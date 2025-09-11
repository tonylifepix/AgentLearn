import os
import time
import json
import argparse
import random
from typing import List

import pyrfume
import numpy as np

import train_rl

try:
    import torch
except Exception:
    raise SystemExit("PyTorch is required to run evaluation. Install with `pip install torch`.")


class SMILESModelEval(torch.nn.Module):
    def __init__(self, vocab_size, embed_size=64, hidden_size=128):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, embed_size)
        self.lstm = torch.nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embed(x)
        out, hidden = self.lstm(emb, hidden)
        logits = self.fc(out)
        return logits, hidden


def generate_sample(model, stoi, itos, device, max_len=120, temperature=1.0, require_valid=False, max_attempts=5):
    model.eval()
    for attempt in range(max_attempts):
        tokens = [stoi['^']]
        h = None
        for _ in range(max_len):
            x = torch.tensor([[tokens[-1]]], dtype=torch.long).to(device)
            logits, h = model(x, h)
            probs = torch.softmax(logits[0, -1] / temperature, dim=-1).cpu().detach().numpy()
            idx = int(np.random.choice(len(probs), p=probs))
            tokens.append(idx)
            if idx == stoi['$']:
                break
        s = ''.join([itos[i] for i in tokens[1:-1]])
        if require_valid and train_rl.RDKit_AVAILABLE:
            if train_rl.is_valid_smiles(s):
                return s
            else:
                continue
        else:
            return s
    return s


def evaluate_checkpoint(ckpt_path: str, n_samples: int = 500, device='cpu', require_valid=True, temperature=1.0):
    # Load dataset and vocab (must match how model was trained)
    smiles = train_rl.load_smiles(limit=2000)
    if len(smiles) == 0:
        raise SystemExit("No SMILES found via pyrfume for evaluation")

    stoi, itos, max_len = train_rl.build_vocab(smiles)
    vocab_size = len(itos)

    # Build model and load checkpoint
    model = SMILESModelEval(vocab_size, embed_size=64, hidden_size=128).to(device)

    # load checkpoint dict
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")
    try:
        d = torch.load(ckpt_path, map_location=device)
    except Exception:
        # maybe pickle
        with open(ckpt_path, 'rb') as f:
            d = train_rl.pickle.load(f)

    model_state = d.get('model_state') if isinstance(d, dict) else None
    if model_state is not None:
        try:
            model.load_state_dict(model_state)
        except Exception:
            # try key nested
            pass

    # prepare dataset fingerprints
    dataset_fps = None
    if train_rl.RDKit_AVAILABLE:
        dataset_fps = train_rl.build_fingerprints(smiles)

    # generate samples
    device = torch.device(device)
    results = []
    valid_count = 0
    qeds = []
    gen_set = set()
    novelty_scores = []
    for i in range(n_samples):
        s = generate_sample(model, stoi, itos, device, max_len=max_len + 5, temperature=temperature, require_valid=require_valid)
        is_valid = train_rl.is_valid_smiles(s) if train_rl.RDKit_AVAILABLE else None
        if is_valid:
            valid_count += 1
        # uniqueness
        gen_set.add(s)
        # qed
        q = train_rl.qed_score(s) if train_rl.RDKit_AVAILABLE else None
        if q is not None and q > 0:
            qeds.append(q)
        # novelty (1 - max tanimoto to dataset)
        if train_rl.RDKit_AVAILABLE and dataset_fps is not None and len(dataset_fps) > 0:
            max_sim = train_rl.max_tanimoto_to_dataset(s, dataset_fps, sample_size=500)
            novelty = 1.0 - max_sim
        else:
            novelty = None
        if novelty is not None:
            novelty_scores.append(novelty)

        results.append({
            'smiles': s,
            'valid': bool(is_valid) if is_valid is not None else None,
            'qed': float(q) if q is not None else None,
            'novelty': float(novelty) if novelty is not None else None,
        })

    # compute metrics
    validity_rate = valid_count / n_samples if n_samples > 0 else 0.0
    uniqueness_rate = len(gen_set) / n_samples if n_samples > 0 else 0.0
    avg_qed = float(np.mean(qeds)) if len(qeds) > 0 else None
    avg_novelty = float(np.mean(novelty_scores)) if len(novelty_scores) > 0 else None

    report = {
        'timestamp': time.strftime('%Y%m%d-%H%M%S'),
        'checkpoint': ckpt_path,
        'n_samples': n_samples,
        'validity_rate': validity_rate,
        'uniqueness_rate': uniqueness_rate,
        'avg_qed': avg_qed,
        'avg_novelty': avg_novelty,
        'samples': results[:min(200, len(results))],
    }

    out_dir = 'reports'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'eval_{os.path.basename(ckpt_path)}_{report["timestamp"]}.json')
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Evaluation written to {out_path}")
    print(f"validity_rate={validity_rate:.3f} uniqueness_rate={uniqueness_rate:.3f} avg_qed={avg_qed} avg_novelty={avg_novelty}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='checkpoints/latest.pt')
    parser.add_argument('--n', type=int, default=500)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--no-require-valid', dest='require_valid', action='store_false')
    parser.add_argument('--temp', type=float, default=1.0)
    args = parser.parse_args()

    evaluate_checkpoint(args.ckpt, n_samples=args.n, device=args.device, require_valid=args.require_valid, temperature=args.temp)


if __name__ == '__main__':
    main()
