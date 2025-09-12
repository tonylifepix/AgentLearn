from init import init
import os
import json
import re
import time
from typing import List, Dict
import pyrfume
import numpy as np
from cerebras.cloud.sdk import Cerebras

from libserach import (
    search_smile_by_description,
    evaluate_smiles_novelty,
    suggest_target_descriptors,
    compute_descriptor_frequencies,
    get_behavior_columns,
    vector_from_descriptors,
)
import train_rl
import subprocess
import shlex
import threading
import io
import re
import selfies as sf
from rdkit import Chem
import random


SMI_TOKEN_RE = re.compile(r"[A-Za-z0-9@+\-\[\]\(\)=#\\/\\]+")


def _extract_smiles(text: str) -> List[str]:
    """Heuristic extraction of SMILES-like tokens from free text."""
    if not text:
        return []
    candidates = SMI_TOKEN_RE.findall(text)
    # filter implausible tokens
    filtered = [c for c in candidates if 2 < len(c) <= 200]
    # dedupe preserving order
    seen = set()
    out = []
    for s in filtered:
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out

# Define the SELFIES alphabet (can be derived from a dataset)
alphabet = sf.get_semantic_robust_alphabet()
alphabet = list(alphabet) # Convert set to list

def generate_random_smiles(max_len=50):
    """
    Generates a random valid SMILES string using SELFIES.
    """
    random_selfies = ""
    # Keep adding random symbols until we reach desired length or an end state
    while len(sf.split_selfies(random_selfies)) < max_len:
        random_selfies += random.choice(alphabet)
        
        # Check if the SELFIES can be decoded. If not, backtrack.
        try:
            smiles = sf.decoder(random_selfies)
            # Use RDKit to perform a final sanity check
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                pass # It's a valid molecule
            else:
                # If RDKit fails, the selfie is likely incomplete/invalid
                random_selfies = random_selfies[:-len(random.choice(alphabet))] # simple backtrack
        except Exception:
            # If decoder fails, remove the last added symbol
             random_selfies = random_selfies[:-len(random.choice(alphabet))] # simple backtrack

    # Final decoding
    try:
        final_smiles = sf.decoder(random_selfies)
        # Final check with RDKit
        if Chem.MolFromSmiles(final_smiles):
            return final_smiles
        else:
            return None # Failed to generate a valid one
    except Exception:
        return None



def _strip_think_sections(text: str) -> str:
    """Remove any <think>...</think> sections some chat models include.

    This helps ensure cleaner JSON parsing and SMILES extraction.
    """
    if not text:
        return text
    try:
        # If there's any closing </think> tag, return only the content after the last one.
        closes = list(re.finditer(r"</think\s*>", text, flags=re.IGNORECASE))
        if closes:
            text = text[closes[-1].end():]
        else:
            # Otherwise strip any <think>...</think> sections if present (safe fallback).
            text = re.sub(r"<think\b[^>]*>.*?</think\s*>", "", text, flags=re.IGNORECASE | re.DOTALL)
    except Exception:
        # If regex fails for any reason, return original text
        return text
    return text.strip()


def generate_novel_smiles(descriptors: List[str], target_n: int = 100, batch_per_call: int = 5, max_iters: int = 1000, run_training: bool = False):
    """Generate novel SMILES with given odor descriptors using the LLM, verify novelty, and save results.

    - descriptors: list of odor terms (must be from leffingwell vocabulary or similar)
    - target_n: desired number of novel molecules to collect
    - batch_per_call: ask the LLM to propose this many SMILES per request
    - max_iters: safety cap on LLM calls
    - run_training: if True, after generation will kick off `train_rl.py` as a subprocess
    """
    init()
    client = Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))

    # prepare storage
    generated: List[Dict] = []
    generated_set = set()
    replay = train_rl.ReplayBuffer(max_size=2000)

    # preload replay with any existing checkpoint (so novelty uses current buffer)
    # attempt to load existing checkpoint so replay buffer gets populated
    ckpt_path = os.path.join("checkpoints", "latest.pt")
    try:
        _ = train_rl.load_checkpoint(ckpt_path, model=None, optimizer=None, replay=replay)
    except Exception:
        pass

    # Build a short system prompt guiding the LLM to output SMILES in JSON array form
    system_msg = {
        "role": "system",
        "content": (
            "You are an assistant that proposes chemically-plausible SMILES strings which have the requested odor properties. "
            "Return only a JSON array of SMILES strings (e.g. [\"CCO\", \"C1=CC=CC=C1\"]). Do not add extra commentary. "
            "Prefer chemically-plausible molecules; suggest stereochemistry when appropriate."
        ),
    }

    # Compute behavior vector for descriptors and include it for precise conditioning
    behavior_cols = get_behavior_columns()
    target_vec = vector_from_descriptors(descriptors)
    target_vec_list = target_vec.tolist() if isinstance(target_vec, np.ndarray) else []

    prompt_template = (
        "Generate {n} distinct SMILES strings whose odor behavior matches the target as closely as possible.\n"
        "Target descriptors: {desc}\n"
        "Behavior vector (length {d} in this exact column order) = {vec}\n"
        "Columns: {cols}\n"
        "Constraints: Prefer valid, synthetically plausible molecules; avoid exact matches in Leffingwell; return a JSON array of SMILES only."
    )

    iter_count = 0
    while len(generated) < target_n and iter_count < max_iters:
        iter_count += 1
        prompt = prompt_template.format(
            n=batch_per_call,
            desc=", ".join(descriptors),
            d=len(behavior_cols),
            vec=json.dumps(target_vec_list),
            cols=json.dumps(behavior_cols),
        )
        messages = [system_msg, {"role": "user", "content": prompt}]

        try:
            resp = client.chat.completions.create(messages=messages, model="qwen-3-235b-a22b-thinking-2507", stream=False)
        except Exception as e:
            print("LLM call failed:", e)
            time.sleep(1.0)
            continue

        # Extract text
        try:
            choice = resp.choices[0].message
            text = "" if choice is None else (choice.content or "")
        except Exception:
            text = ""

        # Strip any hidden/auxiliary "think" sections to avoid jamming JSON parsing
        text = _strip_think_sections(text)

        # Try parse JSON array first
        candidates: List[str] = []
        if text:
            try:
                parsed = json.loads(text)
                # Note: any <think> sections have been stripped above.
                
                if isinstance(parsed, list):
                    candidates = [str(x).strip() for x in parsed if isinstance(x, str)]
            except Exception:
                # fallback to heuristic extraction
                candidates = _extract_smiles(text)

        # If model didn't produce any, skip
        if not candidates:
            continue

        for smi in candidates:
            if len(generated) >= target_n:
                break
            if smi in generated_set:
                continue

            # Validate SMILES using train_rl.is_valid_smiles when available
            valid = False
            try:
                valid = train_rl.is_valid_smiles(smi)
            except Exception:
                # conservative fallback: basic character check
                valid = bool(SMI_TOKEN_RE.fullmatch(smi)) and len(smi) > 2

            # Evaluate novelty relative to Leffingwell dataset
            novelty_info = evaluate_smiles_novelty(smi, top_n=5, similarity_threshold=0.85)

            # Check odor similarity using the search tool (gives nearest molecules for the descriptors)
            odor_matches = []
            try:
                odor_matches = search_smile_by_description(descriptors, top_n=5)
            except Exception:
                odor_matches = []

            novel_flag = False
            try:
                novel_flag = bool(novelty_info.get("novel", False)) and not novelty_info.get("is_in_dataset", False)
            except Exception:
                novel_flag = False

            record = {
                "smiles": smi,
                "valid": bool(valid),
                "novelty": novelty_info,
                "odor_matches": odor_matches,
                "timestamp": time.time(),
            }

            # Accept into generated list only if valid and novel_flag True
            if valid and novel_flag:
                generated.append(record)
                generated_set.add(smi)
                # add fingerprint into replay buffer if RDKit available
                try:
                    fps = train_rl.build_fingerprints([smi])
                    fp = fps[0] if fps else None
                    replay.add(smi, fp)
                except Exception:
                    replay.add(smi, None)
                print(f"Accepted novel SMILES ({len(generated)}/{target_n}): {smi}")
            else:
                # still log the candidate
                print(f"Rejected candidate (valid={valid}, novel={novel_flag}) : {smi}")

        # small sleep to avoid hammering API
        time.sleep(0.2)

    # Save generated list
    out_path = os.environ.get("GENERATED_SMILES_FILE", "generated_smiles.json")
    try:
        with open(out_path, 'w') as f:
            json.dump(generated, f, indent=2)
        print(f"Saved {len(generated)} generated molecules to {out_path}")
    except Exception as e:
        print("Failed to save generated SMILES:", e)

    # Save replay buffer as a lightweight checkpoint so RL can pick it up
    gen_ckpt = os.path.join("checkpoints", "generated_latest.pt")
    try:
        os.makedirs(os.path.dirname(gen_ckpt), exist_ok=True)
        train_rl.save_checkpoint(gen_ckpt, model=None, optimizer=None, replay=replay, step=0)
        print(f"Saved generated replay checkpoint to {gen_ckpt}")
    except Exception as e:
        print("Failed to save generated checkpoint:", e)

    # Optionally run training (this will use train_rl.load code which reads GENERATED_SMILES_FILE)
    if run_training:
        # Set env var so train_rl.load_smiles will include these generated smiles
        env = os.environ.copy()
        env["GENERATED_SMILES_FILE"] = os.path.abspath(out_path)
        # pass target behavior vector and descriptors for RL behavior reward
        try:
            env["TARGET_BEHAVIOR_VECTOR_JSON"] = json.dumps(target_vec_list)
            env["TARGET_DESCRIPTORS"] = ",".join(descriptors)
        except Exception:
            pass
        cmd = f"python {os.path.join(os.getcwd(), 'train_rl.py')}"
        print("Starting RL training subprocess (this will run train_rl.py in a new terminal)")
        try:
            # Launch training in the background so this command returns
            import subprocess
            p = subprocess.Popen(cmd, shell=True, env=env)
            print(f"Launched train_rl.py (pid={p.pid})")
        except Exception as e:
            print("Failed to launch training subprocess:", e)

    return generated


def _run_training_and_capture(env: Dict[str, str], timeout: int = 60 * 60 * 3) -> Dict:
    """Run train_rl.py synchronously with provided env and capture stdout/stderr.

    Returns a dict containing 'stdout' (str), 'stderr' (str), and parsed metrics.
    """
    cmd = [shlex.quote(os.path.join(os.getcwd(), 'train_rl.py'))]
    full_cmd = f"python {cmd[0]}"
    try:
        proc = subprocess.Popen(full_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True)
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "metrics": {}}

    metrics = {}
    # parse avg_reward lines
    avg_rewards = re.findall(r"avg_reward=(\d+\.\d+)", out)
    if avg_rewards:
        try:
            metrics['final_avg_reward'] = float(avg_rewards[-1])
        except Exception:
            pass

    # parse novelty buffer size
    buf_sizes = re.findall(r"novelty_buf=(\d+)", out)
    if buf_sizes:
        try:
            metrics['novelty_buf'] = int(buf_sizes[-1])
        except Exception:
            pass

    # capture example molecules after RL
    examples = []
    m = re.search(r"Example molecules \(after RL\):\n([\s\S]*)", out)
    if m:
        block = m.group(1)
        # stop at next blank line or end
        for line in block.splitlines():
            line = line.strip()
            if not line:
                break
            # likely a SMILES string
            examples.append(line)
    metrics['examples_after_rl'] = examples

    return {"stdout": out, "stderr": err, "metrics": metrics}


def ask_llm_for_adjustments(client: Cerebras, descriptors: List[str], metrics: Dict, recent_generated: List[Dict]) -> Dict:
    """Ask the LLM to suggest next-generation parameters given metrics and examples.

    Expects the model to return a JSON object with optional fields:
      - next_descriptors: [str]
      - batch_per_call: int
      - n_to_generate: int
      - temperature: float
      - direct_smiles: [str]  (optional list of SMILES to try directly)
    """
    system = {
        "role": "system",
        "content": (
            "You are an experiment manager. Given training metrics and example molecules, propose a small set of changes to the next generation pass to improve novelty and validity. "
            "Reply ONLY with a JSON object containing any of: next_descriptors (array of strings), batch_per_call (int), n_to_generate (int), temperature (float), direct_smiles (array of SMILES)."
        )
    }

    user = {
        "role": "user",
        "content": json.dumps({
            "descriptors": descriptors,
            "metrics": metrics,
            "recent_generated": recent_generated[:10]
        })
    }

    try:
        resp = client.chat.completions.create(messages=[system, user], model="qwen-3-235b-a22b-thinking-2507", stream=False)
        choice = resp.choices[0].message
        text = choice.content or ""
        # parse JSON
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            # try to extract JSON substring
            jmatch = re.search(r"(\{[\s\S]*\})", text)
            if jmatch:
                try:
                    return json.loads(jmatch.group(1))
                except Exception:
                    return {}
            return {}
    except Exception:
        return {}


def orchestrate(descriptors: List[str], rounds: int = 3, per_round_target: int = 50):
    """High-level loop: generate -> train -> analyze -> adjust -> repeat.

    - descriptors: initial target descriptors
    - rounds: number of generation-training rounds
    - per_round_target: number of novel molecules to collect per round
    """
    init()
    client = Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))
    all_generated = []

    current_descriptors = descriptors
    # If no descriptors provided, auto-suggest underrepresented descriptor combos
    if not current_descriptors:
        try:
            suggestions = suggest_target_descriptors(n_suggestions=3, comb_max_size=2)
            if suggestions:
                # pick the first suggestion as initial target
                current_descriptors = suggestions[0]
                print(f"Auto-selected target descriptors from dataset analysis: {current_descriptors}")
        except Exception as e:
            print("Failed to auto-select descriptors:", e)
    for r in range(rounds):
        print(f"=== Round {r+1}/{rounds}: descriptors={current_descriptors} ===")
        gen = generate_novel_smiles(current_descriptors, target_n=per_round_target, batch_per_call=5, run_training=False)
        all_generated.extend(gen)

        # write generated to file and set env for training
        out_path = os.path.abspath(os.environ.get("GENERATED_SMILES_FILE", "generated_smiles.json"))
        with open(out_path, 'w') as f:
            json.dump(all_generated, f, indent=2)

        env = os.environ.copy()
        env["GENERATED_SMILES_FILE"] = out_path
        # Provide target behavior vector for RL reward
        try:
            behavior_cols = get_behavior_columns()
            target_vec = vector_from_descriptors(current_descriptors)
            env["TARGET_BEHAVIOR_VECTOR_JSON"] = json.dumps(target_vec.tolist())
            # allow tuning weight via env if set externally; default is defined in train_rl
            if "BEHAVIOR_REWARD_WEIGHT" not in env:
                env["BEHAVIOR_REWARD_WEIGHT"] = "0.3"
        except Exception:
            pass

        print("Running training with newly generated molecules...")
        res = _run_training_and_capture(env)
        print("Training finished; parsing metrics...")
        metrics = res.get('metrics', {})
        print("Parsed metrics:", metrics)

        # Ask LLM for adjustments
        suggestion = ask_llm_for_adjustments(client, current_descriptors, metrics, gen)
        print("LLM suggestion:", suggestion)

        # Apply suggestions
        if 'next_descriptors' in suggestion and isinstance(suggestion['next_descriptors'], list):
            current_descriptors = [str(x) for x in suggestion['next_descriptors'] if x]
        # other numeric suggestions we could pass into generate function next round
        # (not fully wired in this simple prototype)
        # If LLM provided direct_smiles, validate and append them immediately
        if 'direct_smiles' in suggestion and isinstance(suggestion['direct_smiles'], list):
            for s in suggestion['direct_smiles']:
                try:
                    valid = train_rl.is_valid_smiles(s)
                except Exception:
                    valid = False
                nov = evaluate_smiles_novelty(s)
                rec = {"smiles": s, "valid": valid, "novelty": nov, "timestamp": time.time()}
                if valid and nov.get('novel', False):
                    all_generated.append(rec)

    # final save
    final_out = os.environ.get("GENERATED_SMILES_FILE", "generated_smiles.json")
    with open(final_out, 'w') as f:
        json.dump(all_generated, f, indent=2)
    print(f"Orchestration complete. Total generated: {len(all_generated)} saved to {final_out}")



def main():
    # # Example usage: generate 100 novel molecules smelling like floral+vanilla
    # descriptors = os.environ.get("TARGET_DESCRIPTORS", "floral,vanilla").split(',')
    # descriptors = [d.strip() for d in descriptors if d.strip()]
    # target_n = int(os.environ.get("TARGET_N", "100"))
    # run_training = os.environ.get("RUN_TRAINING", "0") in ("1", "true", "True")

    # generated = generate_novel_smiles(descriptors, target_n=target_n, batch_per_call=5, run_training=run_training)
    # # Basic analysis: print summary statistics
    # novel_count = len(generated)
    # print(f"Generation complete: collected {novel_count} novel molecules for descriptors={descriptors}")

    # To run the adaptive orchestration with comparison, call run_controlled_experiment()
    run_controlled_experiment(
        descriptors=['Noctua fans', 'Steam deck'],
        llm_rounds=10,
        per_round_target=20,
        control_n=200,
    )


def _sample_random_control(n: int) -> List[Dict]:
    """Sample n random SMILES from the dataset without any rejection filtering.

    Returns list of records with minimal metadata matching generated format.
    """
    smiles = pyrfume.load_data('leffingwell/molecules.csv')
    if smiles is None or 'IsomericSMILES' not in smiles.columns:
        return []
    s = smiles['IsomericSMILES'].dropna().astype(str).unique().tolist()
    out = []
    for _ in range(n):
        smiles = generate_random_smiles(max_len=30)
        out.append({"smiles": smiles, "valid": train_rl.is_valid_smiles(smiles), "novelty": evaluate_smiles_novelty(smiles), "timestamp": time.time()})
    return out


def _compute_validity_novelty(records: List[Dict]) -> Dict[str, float]:
    """Compute simple validity and novelty metrics from a list of record dicts.

    Each record is expected to have keys:
      - 'valid': bool
      - 'novelty': dict with a boolean-like 'novel' field (fallbacks handled)
    """
    n = len(records)
    valid = sum(1 for r in records if bool(r.get('valid')))
    novel = 0
    for r in records:
        nov = r.get('novelty')
        is_novel = False
        if isinstance(nov, dict):
            # prefer 'novel', but fall back to common alternatives if present
            for k in ('novel', 'is_novel', 'novelty'):
                if k in nov:
                    try:
                        is_novel = bool(nov.get(k))
                    except Exception:
                        is_novel = False
                    break
        novel += 1 if is_novel else 0
    return {
        'n': n,
        'valid': valid,
        'novel': novel,
        'valid_rate': (valid / n) if n else 0.0,
        'novel_rate': (novel / n) if n else 0.0,
    }


def run_controlled_experiment(descriptors: List[str], llm_rounds: int = 2, per_round_target: int = 50, control_n: int = 150):
    """Run a control-group experiment:
      1) Create a pretrained checkpoint (PRETRAIN_ONLY)
      2) Run the LLM-orchestrated pipeline starting from that checkpoint
      3) Run a control pipeline that uses `control_n` random SMILES (no rejection)
      4) Compare final metrics from both runs
    """
    # 1) Create pretrained checkpoint
    pretrain_ckpt = os.path.abspath(os.path.join('checkpoints', 'pretrained.pt'))
    env = os.environ.copy()
    env['PRETRAIN_ONLY'] = '1'
    env['PRETRAIN_SAVE_PATH'] = pretrain_ckpt
    env['NUM_SMILES'] = str(2000)
    print("Creating pretrained checkpoint (this will run train_rl.py PRETRAIN_ONLY)...")
    res = _run_training_and_capture(env, timeout=60*60)
    print("Pretraining stdout (truncated):\n", '\n'.join(res['stdout'].splitlines()[-20:]))

    # 2) LLM-orchestrated run: set SKIP_PRETRAIN so train_rl uses pretrained checkpoint
    print("Starting LLM-orchestrated experiment from pretrained checkpoint...")
    llm_out_path = os.path.abspath('generated_llm.json')
    os.environ['GENERATED_SMILES_FILE'] = llm_out_path
    all_generated = []
    current_desc = descriptors
    client = Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))
    for r in range(llm_rounds):
        gen = generate_novel_smiles(current_desc, target_n=per_round_target, batch_per_call=5, run_training=False)
        all_generated.extend(gen)
        with open(llm_out_path, 'w') as f:
            json.dump(all_generated, f, indent=2)

    env_llm = os.environ.copy()
    env_llm['GENERATED_SMILES_FILE'] = llm_out_path
    env_llm['SKIP_PRETRAIN'] = '1'
    env_llm['PRETRAIN_SAVE_PATH'] = pretrain_ckpt
    # Provide behavior target vector for RL conditioning
    try:
        from libserach import vector_from_descriptors
        tgt_vec = vector_from_descriptors(current_desc)
        env_llm['TARGET_BEHAVIOR_VECTOR_JSON'] = json.dumps(tgt_vec.tolist())
        if 'BEHAVIOR_REWARD_WEIGHT' not in env_llm:
            env_llm['BEHAVIOR_REWARD_WEIGHT'] = '0.3'
    except Exception:
        pass
    # ensure train_rl.load_checkpoint picks up pretrained checkpoint by loading it into latest.pt
    try:
        import shutil
        os.makedirs('checkpoints', exist_ok=True)
        shutil.copy(pretrain_ckpt, os.path.join('checkpoints', 'latest.pt'))
    except Exception:
        pass
    res_llm = _run_training_and_capture(env_llm)
    metrics_llm = res_llm.get('metrics', {})
    print("LLM experiment metrics:", metrics_llm)

    # 3) Control: random sample, no rejection
    print(f"Creating control group of {control_n} random molecules (no rejection)...")
    control_list = _sample_random_control(control_n)
    control_path = os.path.abspath('generated_control.json')
    with open(control_path, 'w') as f:
        json.dump(control_list, f, indent=2)

    env_ctrl = os.environ.copy()
    env_ctrl['GENERATED_SMILES_FILE'] = control_path
    env_ctrl['SKIP_PRETRAIN'] = '1'
    # ensure starting from same pretrained checkpoint
    try:
        shutil.copy(pretrain_ckpt, os.path.join('checkpoints', 'latest.pt'))
    except Exception:
        pass
    res_ctrl = _run_training_and_capture(env_ctrl)
    metrics_ctrl = res_ctrl.get('metrics', {})
    print("Control experiment metrics:", metrics_ctrl)

    # Compute and display validity and novelty for both generated sets
    llm_gen_metrics = _compute_validity_novelty(all_generated)
    ctrl_gen_metrics = _compute_validity_novelty(control_list)
    print("\n=== Generation quality (validity/novelty) ===")
    print(
        "LLM generated: n={n}, valid={valid} ({valid_rate:.1%}), novel={novel} ({novel_rate:.1%})".format(
            **llm_gen_metrics
        )
    )
    print(
        "Control sample: n={n}, valid={valid} ({valid_rate:.1%}), novel={novel} ({novel_rate:.1%})".format(
            **ctrl_gen_metrics
        )
    )

    # 4) Compare
    print("\n=== Comparison ===")
    print("LLM experiment final_avg_reward:", metrics_llm.get('final_avg_reward'))
    print("Control experiment final_avg_reward:", metrics_ctrl.get('final_avg_reward'))
    print("LLM novelty_buf:", metrics_llm.get('novelty_buf'))
    print("Control novelty_buf:", metrics_ctrl.get('novelty_buf'))
    print("LLM examples_after_rl (sample):", metrics_llm.get('examples_after_rl', [])[:5])
    print("Control examples_after_rl (sample):", metrics_ctrl.get('examples_after_rl', [])[:5])

if __name__ == "__main__":
    init()
    main()