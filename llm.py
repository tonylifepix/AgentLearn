from init import init
import os
import json
import re
import time
from typing import List, Dict
import pyrfume
import numpy as np
from cerebras.cloud.sdk import Cerebras

from libserach import search_smile_by_description, evaluate_smiles_novelty
import train_rl
import subprocess
import shlex
import threading
import io
import re


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

    prompt_template = (
        "Generate {n} distinct SMILES strings that likely smell like: {desc}. "
        "Try to propose molecules that are not in the Leffingwell dataset. Return a JSON array of SMILES only."
    )

    iter_count = 0
    while len(generated) < target_n and iter_count < max_iters:
        iter_count += 1
        prompt = prompt_template.format(n=batch_per_call, desc=", ".join(descriptors))
        messages = [system_msg, {"role": "user", "content": prompt}]

        try:
            resp = client.chat.completions.create(messages=messages, model="qwen-3-235b-a22b-thinking-2507")
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

        # Try parse JSON array first
        candidates: List[str] = []
        if text:
            try:
                parsed = json.loads(text)
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
        resp = client.chat.completions.create(messages=[system, user], model="qwen-3-235b-a22b-thinking-2507")
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

    orchestrate(['Noctua fans','Steam deck'], rounds=3, per_round_target=50)

if __name__ == "__main__":
    main()