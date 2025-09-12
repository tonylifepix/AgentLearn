from init import init
import os
import json
import re
import time
from typing import List, Dict, Tuple
import pyrfume
import numpy as np
import pandas as pd
from cerebras.cloud.sdk import Cerebras

from libserach import (
    search_description_by_smiles,
    search_smile_by_description,
    evaluate_smiles_novelty,
    suggest_target_descriptors,
    compute_descriptor_frequencies,
    get_behavior_columns,
    vector_from_descriptors,
)
# Keep RL utilities for validity checks but do not run RL training anymore
import train_rl
import subprocess
import shlex
import threading
import io
import re
import selfies as sf
from rdkit import Chem
import random

# Import predictor components to enable retraining here
from leffingwell_odor_model import (
    Config as PredictorConfig,
    SmilesFeaturizer,
    MLP,
    load_leffingwell_dataframe,
    split_df,
    evaluate as evaluate_predictor,
)


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
    while len(list(sf.split_selfies(random_selfies))) < max_len:
        token = random.choice(alphabet)
        random_selfies += token

        # Check if the SELFIES can be decoded. If not, backtrack.
        try:
            smiles = sf.decoder(random_selfies)
            # Use RDKit to perform a final sanity check
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                pass  # It's a valid molecule
            else:
                # If RDKit fails, the selfie is likely incomplete/invalid
                random_selfies = random_selfies[:-len(token)]  # backtrack last token
        except Exception:
            # If decoder fails, remove the last added symbol
            random_selfies = random_selfies[:-len(token)]  # backtrack last token

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


def generate_novel_smiles(descriptors: List[str], target_n: int = 100, batch_per_call: int = 5, max_iters: int = 1000):
    """Generate novel SMILES using the LLM, verify novelty, label by nearest neighbor, and save results.

    - descriptors: list of odor terms (Leffingwell vocabulary preferred)
    - target_n: desired number of novel molecules to collect
    - batch_per_call: ask the LLM to propose this many SMILES per request
    - max_iters: safety cap on LLM calls
    """
    init()
    client = Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))

    # prepare storage
    generated: List[Dict] = []
    generated_set = set()

    # Build a short system prompt guiding the LLM to output SMILES in JSON array form
    system_msg = {
        "role": "system",
        "content": (
            "You are a chemistry assistant that proposes chemically plausible, synthesizable SMILES strings for molecules that is similar to the most unique molecule in the dataset. "
            "Return only a JSON array of SMILES strings (e.g. [\"CCO\", \"C1=CC=CC=C1\"]). Do not add extra commentary. "
            "Prefer chemically-plausible molecules; suggest stereochemistry when appropriate."
            "You may call tools: "
            "search_smile_by_description(descriptors) to explore known molecules matching the target descriptors;"
            "search_description_by_smiles(smiles) to inspect dataset neighbors of candidates;"
            "Avoid returning any SMILES that exactly match the Leffingwell dataset or are near-duplicates of your own suggestions. If no suitable options, generate another batch and try again."
        ),
    }

    # Compute behavior vector for descriptors and include it for precise conditioning
    behavior_cols = get_behavior_columns()

    prompt_template = (
        "Generate {n} distinct SMILES strings for molecules similar to the most unique molecule. You can use the tools to find the most unique molecule.\n"
        "For each molecules you suggest verify the SMILES,\n"
        "Columns: {cols}\n"
        "Constraints: Prefer valid, synthetically plausible molecules; avoid exact matches in Leffingwell; return a JSON array of SMILES only."
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_smile_by_description",
                "strict": True,
                "description": "A tool that searches for SMILES strings based on valid descriptors for Leffingwell dataset.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "valid descriptors for Leffingwell dataset."
                        }
                    },
                    "required": ["expression"]
                }
            }
        }, 
        {
            "type": "function",
            "function": {
                "name": "search_description_by_smiles",
                "strict": True,
                "description": "A tool that searches for similar molecule and their descriptors based on SMILES strings from Leffingwell dataset.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "smiles": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "valid SMILES strings"
                        }
                    },
                    "required": ["smiles"]
                }
            }
        }
    ]

    # Register every callable tool once
    available_functions = {
        "search_smile_by_description": search_smile_by_description,
        "search_description_by_smiles": search_description_by_smiles
    }

    behavior_cols = get_behavior_columns()
    # Cache datasets for fallback lookups without RDKit
    _mol_df = None
    _beh_df = None

    def _ensure_dfs():
        nonlocal _mol_df, _beh_df
        if _mol_df is None or _beh_df is None:
            try:
                _mol_df = pyrfume.load_data('leffingwell/molecules.csv')
                _beh_df = pyrfume.load_data('leffingwell/behavior.csv')
            except Exception:
                _mol_df, _beh_df = None, None

    def _assign_properties_from_nearest(smi: str, desired_desc: List[str]) -> Tuple[Dict[str, int], List[float]]:
        """Label a SMILES using the odor properties of the most similar dataset molecule.

        Returns (properties_dict, label_vector_aligned_to_behavior_cols).
        If no neighbor found, returns empty dict and zero vector.
        """
        props_dict: Dict[str, int] = {}
        # First try RDKit-based nearest neighbor via novelty helper
        try:
            nov = evaluate_smiles_novelty(smi, top_n=1)
            nearest = (nov or {}).get('nearest') or []
            if nearest:
                cid = nearest[0].get('cid')
                _ensure_dfs()
                if _beh_df is not None and cid in _beh_df.index:
                    row = _beh_df.loc[cid]
                    if hasattr(row, 'iloc') and not isinstance(row, pd.Series):
                        row = row.iloc[0]
                    props_dict = {str(col): int(bool(row[col])) for col in _beh_df.columns}
        except Exception:
            props_dict = {}
        # Fallback: use descriptor-based top match to pick a molecule and use its properties
        if not props_dict:
            try:
                matches = search_smile_by_description(desired_desc or [], top_n=1)
                if matches:
                    smi2 = matches[0].get('IsomericSMILES')
                    _ensure_dfs()
                    if _mol_df is not None and _beh_df is not None and smi2 is not None:
                        # find CID by SMILES string
                        cands = _mol_df.index[_mol_df['IsomericSMILES'] == smi2]
                        if len(cands) > 0:
                            cid = cands[0]
                            if cid in _beh_df.index:
                                row = _beh_df.loc[cid]
                                if hasattr(row, 'iloc') and not isinstance(row, pd.Series):
                                    row = row.iloc[0]
                                props_dict = {str(col): int(bool(row[col])) for col in _beh_df.columns}
            except Exception:
                props_dict = {}

        # build vector
        vec = [float(props_dict.get(col, 0)) for col in behavior_cols]
        return props_dict, vec

    iter_count = 0
    while len(generated) < target_n and iter_count < max_iters:
        iter_count += 1
        prompt = prompt_template.format(
            n=batch_per_call,
            desc=", ".join(descriptors),
            d=len(behavior_cols),
            cols=json.dumps(behavior_cols),
        )
        messages = [system_msg, {"role": "user", "content": prompt}]

        while True:
            try:
                resp = client.chat.completions.create(messages=messages, model="qwen-3-235b-a22b-thinking-2507", stream=False, tools=tools)
            except Exception as e:
                print("LLM call failed:", e)
                time.sleep(2.0)
                continue
            choice = resp.choices[0].message
            if choice.tool_calls:
                function_call = choice.tool_calls[0].function
                if function_call.name == "search_smile_by_description":
                    args = function_call.arguments
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                    descs = args.get("expression") if isinstance(args, dict) else None
                    if isinstance(descs, list):
                        try:
                            tool_result = search_smile_by_description(descs)
                            tool_content = json.dumps(tool_result)
                        except Exception as e:
                            tool_content = json.dumps({"error": str(e)})
                    else:
                        tool_content = json.dumps({"error": "Invalid arguments format"})
                elif function_call.name == "search_description_by_smiles":
                    args = function_call.arguments
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                    smis = args.get("smiles") if isinstance(args, dict) else None
                    if isinstance(smis, list):
                        try:
                            tool_result = search_description_by_smiles(smis)
                            tool_content = json.dumps(tool_result)
                        except Exception as e:
                            tool_content = json.dumps({"error": str(e)})
                    else:
                        tool_content = json.dumps({"error": "Invalid arguments format"})
                messages.append({
                    "role": "tool",
                    "content": tool_content,
                    "tool_call_id": choice.tool_calls[0].id
                })
            else:
                # no tool call/ final_response
                break

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

            # Assign odor labels from nearest neighbor for retraining
            props_dict, label_vec = _assign_properties_from_nearest(smi, descriptors)

            record = {
                "smiles": smi,
                "valid": bool(valid),
                "novelty": novelty_info,
                "odor_matches": odor_matches,
                "assigned_properties": props_dict,
                "assigned_vector": label_vec,
                "timestamp": time.time(),
            }

            # Accept into generated list only if valid and novel_flag True
            if valid and novel_flag:
                generated.append(record)
                generated_set.add(smi)
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

    return generated


def _train_predictor_from_dfs(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame, label_cols: List[str], cfg: PredictorConfig, ckpt_name: str) -> Dict:
    """Train the Leffingwell predictor using provided splits and return metrics/report dict.

    Saves checkpoint and report similarly to leffingwell_odor_model.train_model but on custom splits.
    """
    # Build dataloaders
    feat = SmilesFeaturizer(n_bits=cfg.fingerprint_size, radius=cfg.fingerprint_radius)

    X_train = feat.featurize(df_train['SMILES'].tolist())
    y_train = df_train[label_cols].values.astype(np.float32)
    X_val = feat.featurize(df_val['SMILES'].tolist())
    y_val = df_val[label_cols].values.astype(np.float32)
    X_test = feat.featurize(df_test['SMILES'].tolist())
    y_test = df_test[label_cols].values.astype(np.float32)

    # Minimal in-place dataset to avoid reusing torch Dataset class
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn.functional as F

    dl_train = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    dl_val = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    dl_test = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)), batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # Model
    in_dim = cfg.fingerprint_size
    out_dim = len(label_cols)
    model = MLP(in_dim, out_dim, hidden=cfg.hidden_sizes, dropout=cfg.dropout).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_state = None
    best_val_loss = float('inf')

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        steps = 0
        for xb, yb in dl_train:
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb, reduction='mean')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()
            steps += 1
        train_loss = total / max(steps, 1)
        val_loss, val_f1_micro, val_f1_macro = evaluate_predictor(model, dl_val, cfg.device)
        print(f"[Predictor] Epoch {epoch:03d}/{cfg.epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_f1_micro={val_f1_micro:.4f} val_f1_macro={val_f1_macro:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu() if hasattr(v, 'device') else v for k, v in model.state_dict().items()}

    # Load best state
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final test evaluation
    test_loss, test_f1_micro, test_f1_macro = evaluate_predictor(model, dl_test, cfg.device)

    # Save
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.report_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.ckpt_dir, ckpt_name)
    torch.save({
        'config': cfg.__dict__,
        'model_state': model.state_dict(),
        'n_inputs': in_dim,
        'n_labels': out_dim,
        'arch': {
            'hidden': list(cfg.hidden_sizes),
            'dropout': cfg.dropout,
        },
        'label_names': label_cols,
        'featurizer': {
            'type': 'rdkit_morgan' if feat.use_rdkit else 'ngram_hash',
            'n_bits': feat.n_bits,
            'radius': feat.radius,
        },
        'best_val': {
            'loss': best_val_loss,
        },
    }, ckpt_path)

    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    report = {
        'timestamp': timestamp,
        'train_size': len(df_train),
        'val_size': len(df_val),
        'test_size': len(df_test),
        'test': {
            'loss': test_loss,
            'f1_micro': test_f1_micro,
            'f1_macro': test_f1_macro,
        },
        'ckpt_path': ckpt_path,
    }
    report_path = os.path.join(cfg.report_dir, f"leffingwell_eval_{timestamp}.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    return report


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
            "Reply ONLY with a JSON object containing any of: next_descriptors (array of strings), batch_per_call (int), n_to_generate (int), temperature (float), direct_smiles (array of SMILES). "
            "You have access to two tools: search_smile_by_description and search_description_by_smiles. You may use them to inspect dataset similarities before deciding."
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

    # Define available tools (mirrors generate_novel_smiles)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_smile_by_description",
                "strict": True,
                "description": "A tool that searches for SMILES strings based on valid descriptors for Leffingwell dataset.",
                "parameters": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "valid descriptors for Leffingwell dataset."
                    }
                },
                "required": ["expression"]
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_description_by_smiles",
                "strict": True,
                "description": "A tool that searches for similar molecule and their descriptors based on SMILES strings from Leffingwell dataset.",
                "parameters": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "valid SMILES strings"
                    }
                },
                "required": ["smiles"]
            }
        }
    ]

    messages = [system, user]

    # Multi-turn tool-calling loop similar to generate_novel_smiles
    while True:
        try:
            resp = client.chat.completions.create(
                messages=messages,
                model="qwen-3-235b-a22b-thinking-2507",
                stream=False,
                tools=tools,
            )
        except Exception:
            # On API failure, return empty suggestion
            return {}

        choice = getattr(resp.choices[0], 'message', None)
        if not choice:
            return {}

        # If model requested a tool, execute it and continue the loop
        if getattr(choice, 'tool_calls', None):
            try:
                call = choice.tool_calls[0]
                fn = call.function
                tool_content = json.dumps({"error": "Unknown tool"})
                if fn.name == "search_smile_by_description":
                    args = getattr(fn, 'arguments', {})
                    descs = args.get("expression") if isinstance(args, dict) else None
                    if isinstance(descs, list):
                        try:
                            result = search_smile_by_description(descs)
                            tool_content = json.dumps(result)
                        except Exception as e:
                            tool_content = json.dumps({"error": str(e)})
                    else:
                        tool_content = json.dumps({"error": "Invalid arguments format"})
                elif fn.name == "search_description_by_smiles":
                    args = getattr(fn, 'arguments', {})
                    smis = args.get("smiles") if isinstance(args, dict) else None
                    if isinstance(smis, list):
                        try:
                            result = search_description_by_smiles(smis)
                            tool_content = json.dumps(result)
                        except Exception as e:
                            tool_content = json.dumps({"error": str(e)})
                    else:
                        tool_content = json.dumps({"error": "Invalid arguments format"})

                messages.append({
                    "role": "tool",
                    "content": tool_content,
                    "tool_call_id": call.id,
                })
                # continue loop for another assistant turn
                continue
            except Exception:
                # Fallback: break and try to parse whatever content exists
                pass

        # No tool call => final content to parse
        text = choice.content or ""
        text = _strip_think_sections(text)
        if not text:
            return {}
        # Prefer direct JSON object
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            # Try to extract JSON object substring
            jmatch = re.search(r"(\{[\s\S]*\})", text)
            if jmatch:
                try:
                    obj = json.loads(jmatch.group(1))
                    return obj if isinstance(obj, dict) else {}
                except Exception:
                    return {}
            return {}

    # Safety fallback
    return {}


def orchestrate(descriptors: List[str], rounds: int = 3, per_round_target: int = 50):
    """High-level loop: generate -> retrain predictor -> analyze -> adjust -> repeat.

    This now augments and retrains the Leffingwell predictor instead of RL generation.
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
        gen = generate_novel_smiles(current_descriptors, target_n=per_round_target, batch_per_call=5)
        all_generated.extend(gen)

        # write generated to file
        out_path = os.path.abspath(os.environ.get("GENERATED_SMILES_FILE", "generated_smiles.json"))
        with open(out_path, 'w') as f:
            json.dump(all_generated, f, indent=2)

        # Retrain predictor on augmented data and evaluate
        print("Retraining predictor with newly generated molecules...")
        try:
            base_df, label_cols = load_leffingwell_dataframe()
            # Build generated df rows with labels
            rows = []
            for rec in gen:
                if not rec.get('valid'):
                    continue
                vec = rec.get('assigned_vector') or []
                if not vec or len(vec) != len(label_cols):
                    continue
                row = {'SMILES': rec['smiles']}
                row.update({c: int(v >= 0.5) for c, v in zip(label_cols, vec)})
                rows.append(row)
            gen_df = pd.DataFrame(rows)
            # Use the same split seed and holdout; augment the training split only
            df_train, df_val, df_test = split_df(base_df, train_ratio=0.8, val_ratio=0.1)
            df_train_aug = pd.concat([df_train, gen_df], ignore_index=True) if not gen_df.empty else df_train
            cfg = PredictorConfig(epochs=5, ckpt_name='leffingwell_augmented.pt')
            report = _train_predictor_from_dfs(df_train_aug, df_val, df_test, label_cols, cfg, ckpt_name=cfg.ckpt_name)
            metrics = report.get('test', {})
        except Exception as e:
            print("Predictor retraining failed:", e)
            metrics = {}
        print("Predictor metrics:", metrics)

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
    smiles_df = pyrfume.load_data('leffingwell/molecules.csv')
    if smiles_df is None or 'IsomericSMILES' not in smiles_df.columns:
        return []
    candidates = smiles_df['IsomericSMILES'].dropna().astype(str).unique().tolist()
    import random
    sampled = random.sample(candidates, min(n, len(candidates)))
    out = []
    for smi in sampled:
        out.append({
            "smiles": smi,
            "valid": train_rl.is_valid_smiles(smi),
            "novelty": evaluate_smiles_novelty(smi),
            "timestamp": time.time()
        })
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
    """Control-group experiment for predictor retraining:
      1) Train a baseline predictor on the Leffingwell dataset (and record test F1)
      2) Generate molecules with LLM, label by nearest neighbor, augment training set, retrain predictor
      3) Control: sample random molecules from dataset, label by nearest neighbor (trivial), augment, retrain
      4) Compare predictor test metrics between LLM-augmented and control-augmented runs
    """
    # Baseline split
    base_df, label_cols = load_leffingwell_dataframe()

    def _robust_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        t, v, te = split_df(df, train_ratio=0.8, val_ratio=0.1)
        # Ensure val/test have at least 1 row if possible
        remaining = df
        if len(v) == 0 and len(df) > 2:
            v = df.sample(n=1, random_state=42)
        if len(te) == 0 and len(df) > 2:
            te = df.drop(v.index).sample(n=1, random_state=43) if len(df.drop(v.index)) > 0 else v
        # Rebuild train as the rest
        t = df.drop(v.index.union(te.index)) if len(v) + len(te) < len(df) else df
        return t.reset_index(drop=True), v.reset_index(drop=True), te.reset_index(drop=True)

    df_train, df_val, df_test = _robust_split(base_df)

    print("Training baseline predictor...")
    base_cfg = PredictorConfig(epochs=1, ckpt_name='leffingwell_base.pt')
    base_report = _train_predictor_from_dfs(df_train, df_val, df_test, label_cols, base_cfg, ckpt_name=base_cfg.ckpt_name)
    print("Baseline test:", base_report.get('test', {}))

    # LLM-augmented
    print("Generating LLM-augmented molecules...")
    llm_out_path = os.path.abspath('generated_llm.json')
    all_generated = []
    current_desc = descriptors
    client = Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))
    for r in range(llm_rounds):
        gen = generate_novel_smiles(current_desc, target_n=per_round_target, batch_per_call=5)
        all_generated.extend(gen)
        with open(llm_out_path, 'w') as f:
            json.dump(all_generated, f, indent=2)

    # Build augmentation df from generated
    gen_rows = []
    for rec in all_generated:
        if not rec.get('valid'):
            continue
        vec = rec.get('assigned_vector') or []
        if not vec or len(vec) != len(label_cols):
            continue
        row = {'SMILES': rec['smiles']}
        row.update({c: int(v >= 0.5) for c, v in zip(label_cols, vec)})
        gen_rows.append(row)
    gen_df = pd.DataFrame(gen_rows)
    df_train_llm = pd.concat([df_train, gen_df], ignore_index=True) if not gen_df.empty else df_train
    print(f"LLM augmentation size: {len(gen_df)}")
    llm_cfg = PredictorConfig(epochs=1, ckpt_name='leffingwell_llm_aug.pt')
    report_llm = _train_predictor_from_dfs(df_train_llm, df_val, df_test, label_cols, llm_cfg, ckpt_name=llm_cfg.ckpt_name)
    metrics_llm = report_llm.get('test', {})
    print("LLM-augmented test:", metrics_llm)

    # Control augmentation: random molecules (no rejection)
    print(f"Creating control group of {control_n} random molecules (no rejection)...")
    control_list = _sample_random_control(control_n)
    control_path = os.path.abspath('generated_control.json')
    with open(control_path, 'w') as f:
        json.dump(control_list, f, indent=2)

    # Label control molecules via nearest neighbor too (though from dataset, it will mirror true labels)
    ctrl_rows = []
    for rec in control_list:
        smi = rec['smiles']
        try:
            nn = search_description_by_smiles([smi], top_n=1)
            props = (nn[0].get('properties') if nn else {}) or {}
            row = {'SMILES': smi}
            row.update({c: int(bool(props.get(c, 0))) for c in label_cols})
            ctrl_rows.append(row)
        except Exception:
            continue
    ctrl_df = pd.DataFrame(ctrl_rows)
    df_train_ctrl = pd.concat([df_train, ctrl_df], ignore_index=True) if not ctrl_df.empty else df_train
    print(f"Control augmentation size: {len(ctrl_df)}")
    ctrl_cfg = PredictorConfig(epochs=1, ckpt_name='leffingwell_ctrl_aug.pt')
    report_ctrl = _train_predictor_from_dfs(df_train_ctrl, df_val, df_test, label_cols, ctrl_cfg, ckpt_name=ctrl_cfg.ckpt_name)
    metrics_ctrl = report_ctrl.get('test', {})
    print("Control-augmented test:", metrics_ctrl)

    # Quality of generated molecules
    llm_gen_metrics = _compute_validity_novelty(all_generated)
    ctrl_gen_metrics = _compute_validity_novelty(control_list)
    print("\n=== Generation quality (validity/novelty) ===")
    print("LLM generated: n={n}, valid={valid} ({valid_rate:.1%}), novel={novel} ({novel_rate:.1%})".format(**llm_gen_metrics))
    print("Control sample: n={n}, valid={valid} ({valid_rate:.1%}), novel={novel} ({novel_rate:.1%})".format(**ctrl_gen_metrics))

    # Comparison
    print("\n=== Predictor comparison ===")
    print("Baseline f1_micro:", base_report.get('test', {}).get('f1_micro'))
    print("LLM-augmented f1_micro:", metrics_llm.get('f1_micro'))
    print("Control-augmented f1_micro:", metrics_ctrl.get('f1_micro'))

if __name__ == "__main__":
    init()
    main()