import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import pyrfume  # type: ignore

# Optional RDKit featurizer. If unavailable, we'll fall back to a simple n-gram hash.
try:
    from rdkit import Chem
    from rdkit import DataStructs
    from rdkit.Chem import AllChem
    _HAS_RDKIT = True
except Exception:
    _HAS_RDKIT = False


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


@dataclass
class Config:
    batch_size: int = 256
    epochs: int = 15
    lr: float = 1e-3
    weight_decay: float = 1e-5
    hidden_sizes: Tuple[int, int] = (1024, 512)
    dropout: float = 0.2
    fingerprint_size: int = 2048
    fingerprint_radius: int = 2
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    report_dir: str = "reports"
    ckpt_dir: str = "checkpoints"
    ckpt_name: str = "leffingwell_latest.pt"


class SmilesFeaturizer:
    """
    Converts SMILES to fixed-length feature vectors.
    - If RDKit is available: use Morgan fingerprints (ECFP) as binary vectors.
    - Else: character n-gram hashing (n in {2,3}) into a fixed-size vector.
    """

    def __init__(self, n_bits: int = 2048, radius: int = 2):
        self.n_bits = n_bits
        self.radius = radius
        self.use_rdkit = _HAS_RDKIT

    def _rdkit_fp(self, smi: str) -> Optional[np.ndarray]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
        arr = np.zeros((self.n_bits,), dtype=np.float32)
        # RDKit fills a list of bit indices; convert to numpy array
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    @staticmethod
    def _ngrams(s: str, n: int) -> List[str]:
        return [s[i : i + n] for i in range(max(len(s) - n + 1, 0))]

    def _hash_features(self, smi: str) -> np.ndarray:
        # Simple signed hashing of 2-grams and 3-grams into n_bits-length vector
        vec = np.zeros((self.n_bits,), dtype=np.float32)
        for n in (2, 3):
            for g in self._ngrams(smi, n):
                # Python's hash is salted per-process; use a stable hash
                h = self._stable_hash(g)
                idx = h % self.n_bits
                sign = 1.0 if ((h >> 31) & 1) == 0 else -1.0
                vec[idx] += sign
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    @staticmethod
    def _stable_hash(s: str) -> int:
        # FNV-1a 32-bit hash for stability
        h = 0x811C9DC5
        for c in s.encode("utf-8"):
            h ^= c
            h = (h * 0x01000193) & 0xFFFFFFFF
        return int(h)

    def featurize(self, smiles: List[str]) -> np.ndarray:
        X = []
        for smi in smiles:
            if smi is None or not isinstance(smi, str) or len(smi.strip()) == 0:
                X.append(np.zeros((self.n_bits,), dtype=np.float32))
                continue
            if self.use_rdkit:
                arr = self._rdkit_fp(smi)
                if arr is None:
                    arr = self._hash_features(smi)
            else:
                arr = self._hash_features(smi)
            X.append(arr.astype(np.float32))
        return np.vstack(X)


class LeffingwellDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: Tuple[int, int] = (1024, 512), dropout: float = 0.2):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def multilabel_f1(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Tuple[float, float]:
    """
    Compute micro and macro F1 scores at a fixed threshold (no sklearn dependency).
    y_true: (N, C) binary
    y_prob: (N, C) probabilities in [0,1]
    """
    y_pred = (y_prob >= threshold).astype(np.int32)
    eps = 1e-8

    # Micro
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    prec_micro = tp / (tp + fp + eps)
    rec_micro = tp / (tp + fn + eps)
    f1_micro = 2 * prec_micro * rec_micro / (prec_micro + rec_micro + eps)

    # Macro
    f1s = []
    for c in range(y_true.shape[1]):
        yt = y_true[:, c]
        yp = y_pred[:, c]
        tp_c = np.sum((yt == 1) & (yp == 1))
        fp_c = np.sum((yt == 0) & (yp == 1))
        fn_c = np.sum((yt == 1) & (yp == 0))
        prec_c = tp_c / (tp_c + fp_c + eps)
        rec_c = tp_c / (tp_c + fn_c + eps)
        f1_c = 2 * prec_c * rec_c / (prec_c + rec_c + eps)
        f1s.append(f1_c)
    f1_macro = float(np.mean(f1s))

    return float(f1_micro), f1_macro


def load_leffingwell_dataframe() -> Tuple[pd.DataFrame, List[str]]:
    """Load Leffingwell behavior and map to SMILES.

    Returns:
        df: DataFrame with columns [CID, SMILES, <113 labels>]
        label_cols: list of label column names (length 113)
    """
    molecules = pyrfume.load_data('leffingwell/molecules.csv').reset_index()
    behavior = pyrfume.load_data('leffingwell/behavior.csv').reset_index()
    behavior.columns = behavior.columns.map(lambda c: str(c).strip())
    molecules.columns = molecules.columns.map(lambda c: str(c).strip())
    label_cols = [c for c in behavior.columns if c != 'Stimulus']
    merged = pd.merge(molecules[['CID', 'IsomericSMILES']], behavior, left_on='CID', right_on='Stimulus').drop(columns='Stimulus').rename(columns={"IsomericSMILES": "SMILES"})
    #merged = merged[merged['IsomericSMILES'].astype(str).str.len() > 0]
    merged[label_cols] = merged[label_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    merged[label_cols] = (merged[label_cols] > 0).astype(np.float32)
    if merged.empty:
        raise RuntimeError("Merged Leffingwell dataframe is empty after joining behavior with stimuli/molecules. Check dataset availability.")
    return merged, label_cols


def split_df(df: pd.DataFrame, train_ratio=0.8, val_ratio=0.1, seed=RANDOM_SEED) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    idx = np.arange(len(df))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def build_dataloaders(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame, label_cols: List[str], cfg: Config):
    feat = SmilesFeaturizer(n_bits=cfg.fingerprint_size, radius=cfg.fingerprint_radius)

    X_train = feat.featurize(df_train['SMILES'].tolist())
    y_train = df_train[label_cols].values.astype(np.float32)
    X_val = feat.featurize(df_val['SMILES'].tolist())
    y_val = df_val[label_cols].values.astype(np.float32)
    X_test = feat.featurize(df_test['SMILES'].tolist())
    y_test = df_test[label_cols].values.astype(np.float32)

    ds_train = LeffingwellDataset(X_train, y_train)
    ds_val = LeffingwellDataset(X_val, y_val)
    ds_test = LeffingwellDataset(X_test, y_test)

    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    dl_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    return dl_train, dl_val, dl_test, feat


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, float, float]:
    model.eval()
    losses = []
    ys = []
    ps = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = F.binary_cross_entropy_with_logits(logits, yb, reduction='mean')
        probs = torch.sigmoid(logits)
        losses.append(loss.item())
        ys.append(yb.detach().cpu().numpy())
        ps.append(probs.detach().cpu().numpy())
    y_true = np.vstack(ys)
    y_prob = np.vstack(ps)
    f1_micro, f1_macro = multilabel_f1(y_true, y_prob)
    return float(np.mean(losses)), f1_micro, f1_macro


def train_model(cfg: Config) -> dict:
    df, label_cols = load_leffingwell_dataframe()
    df_train, df_val, df_test = split_df(df, train_ratio=0.8, val_ratio=0.1, seed=RANDOM_SEED)
    dl_train, dl_val, dl_test, featurizer = build_dataloaders(df_train, df_val, df_test, label_cols, cfg)

    in_dim = cfg.fingerprint_size
    out_dim = len(label_cols)
    model = MLP(in_dim, out_dim, hidden=cfg.hidden_sizes, dropout=cfg.dropout).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = {
        'loss': float('inf'),
        'f1_micro': 0.0,
        'f1_macro': 0.0,
        'epoch': -1,
        'state_dict': None,
    }

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for xb, yb in dl_train:
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb, reduction='mean')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)
        val_loss, val_f1_micro, val_f1_macro = evaluate(model, dl_val, cfg.device)
        print(f"Epoch {epoch:03d}/{cfg.epochs} - train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}  val_f1_micro: {val_f1_micro:.4f}  val_f1_macro: {val_f1_macro:.4f}")

        if val_loss < best_val['loss']:
            best_val.update({
                'loss': val_loss,
                'f1_micro': val_f1_micro,
                'f1_macro': val_f1_macro,
                'epoch': epoch,
                'state_dict': model.state_dict(),
            })

    # Load best
    if best_val['state_dict'] is not None:
        model.load_state_dict(best_val['state_dict'])

    test_loss, test_f1_micro, test_f1_macro = evaluate(model, dl_test, cfg.device)
    print(f"Test - loss: {test_loss:.4f}  f1_micro: {test_f1_micro:.4f}  f1_macro: {test_f1_macro:.4f}")

    # Save checkpoint and report
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.report_dir, exist_ok=True)

    ckpt_path = os.path.join(cfg.ckpt_dir, cfg.ckpt_name)
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
            'type': 'rdkit_morgan' if featurizer.use_rdkit else 'ngram_hash',
            'n_bits': featurizer.n_bits,
            'radius': featurizer.radius,
        },
        'best_val': {k: (v if k != 'state_dict' else None) for k, v in best_val.items()},
    }, ckpt_path)

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    report = {
        'timestamp': timestamp,
        'train_size': len(dl_train.dataset),
        'val_size': len(dl_val.dataset),
        'test_size': len(dl_test.dataset),
        'best_val': {k: (v if k != 'state_dict' else None) for k, v in best_val.items()},
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


@torch.no_grad()
def predict_smiles(smiles: List[str], ckpt_path: str) -> Tuple[np.ndarray, List[str]]:
    """Load a saved checkpoint and predict probabilities for a batch of SMILES.

    Returns: (probs: np.ndarray [N, 113], label_names: List[str])
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device)
    n_inputs = ckpt['n_inputs']
    n_labels = ckpt['n_labels']
    label_names = ckpt.get('label_names', [f"label_{i}" for i in range(n_labels)])
    feat_cfg = ckpt['featurizer']

    feat = SmilesFeaturizer(n_bits=feat_cfg.get('n_bits', 2048), radius=feat_cfg.get('radius', 2))
    # Do not force RDKit if the checkpoint used hashing originally
    if feat_cfg.get('type') == 'ngram_hash':
        feat.use_rdkit = False

    X = feat.featurize(smiles)
    arch = ckpt.get('arch', {})
    hidden = tuple(arch.get('hidden', (1024, 512)))
    dropout = float(arch.get('dropout', 0.2))
    model = MLP(n_inputs, n_labels, hidden=hidden, dropout=dropout)
    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()
    xb = torch.from_numpy(X).to(device)
    logits = model(xb)
    probs = torch.sigmoid(logits).cpu().numpy()
    return probs, label_names


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/test a model to predict 113 Leffingwell odor descriptors from SMILES.")
    sub = p.add_subparsers(dest='cmd', required=False)

    # Train
    p.add_argument('--train', action='store_true', help='Run training (shortcut for cmd=train)')
    p.add_argument('--epochs', type=int, default=Config.epochs)
    p.add_argument('--batch-size', type=int, default=Config.batch_size)
    p.add_argument('--lr', type=float, default=Config.lr)
    p.add_argument('--weight-decay', type=float, default=Config.weight_decay)
    p.add_argument('--dropout', type=float, default=Config.dropout)
    p.add_argument('--hidden', type=str, default=f"{Config.hidden_sizes[0]},{Config.hidden_sizes[1]}", help='Comma-separated hidden sizes')
    p.add_argument('--fp-size', type=int, default=Config.fingerprint_size)
    p.add_argument('--fp-radius', type=int, default=Config.fingerprint_radius)
    p.add_argument('--ckpt', type=str, default=os.path.join(Config.ckpt_dir, Config.ckpt_name))
    p.add_argument('--dry-run', action='store_true', help='Load data and print shapes, then exit')

    # Predict
    p.add_argument('--predict', type=str, nargs='*', help='Predict for provided SMILES strings (space-separated)')
    p.add_argument('--predict-file', type=str, help='Path to a file with one SMILES per line to predict')

    return p.parse_args()


def main():
    args = parse_args()

    hidden = tuple(int(x) for x in args.hidden.split(',')) if isinstance(args.hidden, str) else Config.hidden_sizes
    cfg = Config(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_sizes=hidden,  # type: ignore
        dropout=args.dropout,
        fingerprint_size=args.fp_size,
        fingerprint_radius=args.fp_radius,
        ckpt_name=os.path.basename(args.ckpt),
    )

    if args.dry_run:
        df, label_cols = load_leffingwell_dataframe()
        df_train, df_val, df_test = split_df(df)
        feat = SmilesFeaturizer(n_bits=cfg.fingerprint_size, radius=cfg.fingerprint_radius)
        X_sample = feat.featurize(df_train['SMILES'].head(8).tolist())
        print(f"Loaded Leffingwell: total={len(df)} train={len(df_train)} val={len(df_val)} test={len(df_test)}")
        print(f"Feature dim={X_sample.shape[1]}, labels={len(label_cols)} (expected 113)")
        return

    if args.predict or args.predict_file:
        smiles_list = []
        if args.predict:
            smiles_list.extend(args.predict)
        if args.predict_file:
            with open(args.predict_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        smiles_list.append(line)
        if not smiles_list:
            print("No SMILES provided for prediction.")
            return
        ckpt_path = args.ckpt
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path}")
            return
        probs, label_names = predict_smiles(smiles_list, ckpt_path)
        out = []
        for smi, p in zip(smiles_list, probs):
            out.append({"smiles": smi, "pred": {name: float(prob) for name, prob in zip(label_names, p.tolist())}})
        print(json.dumps(out, indent=2))
        return

    # Default: train
    report = train_model(cfg)
    print("Saved:")
    print(f"- checkpoint: {report['ckpt_path']}")
    # Try to reflect in reports/ for discoverability
    print(f"- report: {os.path.join(cfg.report_dir, 'leffingwell_eval_' + report['timestamp'] + '.json')}")


if __name__ == "__main__":
    main()
