import pyrfume
import numpy as np
import random
import time
from typing import List, Tuple
import difflib
import os
import pickle
import json

try:
    from rdkit import Chem
    from rdkit.Chem import QED
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
    RDKit_AVAILABLE = True
except Exception:
    RDKit_AVAILABLE = False

if RDKit_AVAILABLE:
    try:
        # silence RDKit parser warnings which are noisy during batch parsing
        from rdkit import RDLogger

        RDLogger.DisableLog('rdApp.*')
    except Exception:
        pass


def load_smiles(limit: int = None) -> List[str]:
    """Load IsomericSMILES from leffingwell/molecules.csv using pyrfume.

    Returns a list of SMILES strings (non-null). If `limit` is provided,
    returns at most `limit` smiles (randomly sampled).
    """
    mol = pyrfume.load_data("leffingwell/molecules.csv")
    if mol is None or "IsomericSMILES" not in mol.columns:
        return []
    s = mol["IsomericSMILES"].dropna().astype(str).unique().tolist()
    # Optionally extend dataset with externally-generated SMILES provided via
    # the environment variable GENERATED_SMILES_FILE. The file may be JSON
    # (list of SMILES) or plain text with one SMILES per line.
    try:
        gen_path = os.environ.get("GENERATED_SMILES_FILE")
        if gen_path and os.path.exists(gen_path):
            try:
                with open(gen_path, 'r') as f:
                    data = f.read()
                try:
                    extra = json.loads(data)
                    if isinstance(extra, str):
                        extra = [extra]
                    extra = [str(x).strip() for x in extra if x]
                except Exception:
                    # fallback to newline-split
                    extra = [l.strip() for l in data.splitlines() if l.strip()]

                # append unique extras not already in dataset
                for smi in extra:
                    if smi not in s:
                        s.append(smi)
                if len(extra) > 0:
                    print(f"Loaded {len(extra)} extra generated SMILES from {gen_path}")
            except Exception:
                # don't fail dataset loading if extra file is malformed
                pass
    except Exception:
        pass
    if limit is not None and len(s) > limit:
        return random.sample(s, limit)
    return s


def build_vocab(smiles: List[str]) -> Tuple[dict, dict, int]:
    """Build character vocabulary for SMILES. Adds start '^' and end '$' tokens.

    Returns (stoi, itos, max_len)
    """
    chars = set()
    max_len = 0
    for smi in smiles:
        chars.update(list(smi))
        max_len = max(max_len, len(smi))
    chars = sorted(list(chars))
    # Reserve tokens
    toks = ["^", "$"] + chars
    itos = toks
    stoi = {c: i for i, c in enumerate(itos)}
    return stoi, itos, max_len + 1  # +1 for end token


def max_similarity_to_dataset(smi: str, dataset: List[str], sample_size: int = 500) -> float:
    """Compute maximum sequence similarity (difflib ratio) to a (sample of) dataset.

    This produces a heuristic reward in [0,1]. Using a sample keeps it fast.
    """
    if len(dataset) == 0:
        return 0.0
    if len(dataset) > sample_size:
        candidates = random.sample(dataset, sample_size)
    else:
        candidates = dataset
    best = 0.0
    for t in candidates:
        r = difflib.SequenceMatcher(None, smi, t).ratio()
        if r > best:
            best = r
            if best >= 0.999:
                break
    return best


def build_fingerprints(smiles: List[str], radius: int = 2, nbits: int = 2048):
    """Build Morgan fingerprints for a list of SMILES. Returns list of RDKit ExplicitBitVect or [] if none.

    Invalid or unparsable SMILES are skipped.
    """
    if not RDKit_AVAILABLE:
        return []
    fps = []
    for s in smiles:
        try:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
            fps.append(fp)
        except Exception:
            continue
    return fps


def max_tanimoto_to_dataset(smi: str, dataset_fps: List, sample_size: int = 500) -> float:
    """Compute maximum Tanimoto similarity between SMILES `smi` and a list of
    dataset fingerprints (RDKit ExplicitBitVect). Returns 0.0 if cannot parse or no fps.
    """
    if not RDKit_AVAILABLE:
        return 0.0
    if dataset_fps is None or len(dataset_fps) == 0:
        return 0.0
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return 0.0
        fp_q = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=dataset_fps[0].GetNumBits())
    except Exception:
        return 0.0

    # sample dataset fps for speed
    candidates = dataset_fps if len(dataset_fps) <= sample_size else random.sample(dataset_fps, sample_size)
    best = 0.0
    for fp in candidates:
        try:
            sim = DataStructs.TanimotoSimilarity(fp_q, fp)
            if sim > best:
                best = sim
                if best >= 0.999:
                    break
        except Exception:
            continue
    return float(best)


class ReplayBuffer:
    """Simple FIFO replay buffer storing valid SMILES and their fingerprints."""

    def __init__(self, max_size: int = 1000):
        self.max_size = int(max_size)
        self.smiles = []
        self.fps = []

    def add(self, smi: str, fp):
        if smi is None or fp is None:
            return
        self.smiles.append(smi)
        self.fps.append(fp)
        if len(self.smiles) > self.max_size:
            # pop oldest
            self.smiles.pop(0)
            self.fps.pop(0)

    def __len__(self):
        return len(self.smiles)


def save_checkpoint(path: str, model, optimizer, replay: ReplayBuffer, step: int = None):
    """Save model state, optimizer state, replay buffer smiles, and step to path.

    Uses torch.save if torch is available; falls back to pickle for non-torch parts.
    """
    d = {}
    # model and optimizer state_dicts
    try:
        import torch
        d['model_state'] = model.state_dict()
        d['optimizer_state'] = optimizer.state_dict() if optimizer is not None else None
    except Exception:
        # unlikely in training mode, but continue
        d['model_state'] = None
        d['optimizer_state'] = None

    # replay buffer: store SMILES only and rebuild fps on load
    d['replay_smiles'] = replay.smiles if replay is not None else []
    d['step'] = int(step) if step is not None else None

    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        # prefer torch.save for better compatibility with tensors
        import torch
        torch.save(d, path)
    except Exception:
        # fallback to pickle
        with open(path, 'wb') as f:
            pickle.dump(d, f)


def load_checkpoint(path: str, model, optimizer, replay: ReplayBuffer):
    """Load checkpoint from path, restore model/optimizer state and replay buffer.

    Returns loaded_step or None.
    """
    if not os.path.exists(path):
        return None
    try:
        import torch
        d = torch.load(path, map_location='cpu')
    except Exception:
        with open(path, 'rb') as f:
            d = pickle.load(f)

    # restore model/optimizer
    try:
        if d.get('model_state') is not None:
            model.load_state_dict(d['model_state'])
        if optimizer is not None and d.get('optimizer_state') is not None:
            optimizer.load_state_dict(d['optimizer_state'])
    except Exception:
        pass

    # rebuild replay buffer from smiles
    try:
        saved_smiles = d.get('replay_smiles', [])
        if saved_smiles and replay is not None:
            # clear existing
            replay.smiles = []
            replay.fps = []
            # build fps and add
            fps = build_fingerprints(saved_smiles)
            for s, fp in zip(saved_smiles, fps):
                replay.add(s, fp)
            # if build_fingerprints returned fewer fps than smiles, add remaining smiles without fps
            if len(fps) < len(saved_smiles):
                for s in saved_smiles[len(fps):]:
                    replay.add(s, None)
    except Exception:
        pass

    return d.get('step')


def compute_novelty_score(smi: str, buffer: ReplayBuffer, sample_size: int = 200) -> float:
    """Compute novelty score in [0,1] relative to the replay buffer.

    novelty = 1 - max_tanimoto_to_dataset(smi, buffer.fps, sample_size=sample_size)
    If buffer is empty, return 1.0 (max novelty).
    """
    if buffer is None or len(buffer) == 0:
        return 1.0
    try:
        max_sim = max_tanimoto_to_dataset(smi, buffer.fps, sample_size=sample_size)
        return float(max(0.0, min(1.0, 1.0 - max_sim)))
    except Exception:
        return 0.0


def is_valid_smiles(smi: str) -> bool:
    """Return True if RDKit can parse the SMILES and it yields a chemically-valid molecule.

    If RDKit isn't available, conservatively return False to indicate unknown validity.
    """
    if not RDKit_AVAILABLE:
        return False
    if smi is None or len(smi) == 0:
        return False
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return False
        # Optional: try sanitization, but MolFromSmiles usually sanitizes
        return True
    except Exception:
        return False


def qed_score(smi: str) -> float:
    """Return RDKit QED score in [0,1] if available, else 0.0."""
    if not RDKit_AVAILABLE:
        return 0.0
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return 0.0
        return float(QED.qed(mol))
    except Exception:
        return 0.0


def compute_reward(smi: str, dataset: List[str], sample_size: int = 500) -> float:
    """Combined reward using validity, qed, and similarity heuristics.

    If RDKit is available, reward = 0.4*valid + 0.4*qed + 0.2*similarity
    If RDKit not available, fallback to similarity-only reward.
    All outputs in [0,1].
    """
    # use fingerprint/Tanimoto similarity when available for chemically-meaningful
    # similarity scores
    sim = 0.0
    try:
        if RDKit_AVAILABLE and 'dataset_fps' in globals() and globals().get('dataset_fps') is not None:
            sim = max_tanimoto_to_dataset(smi, globals().get('dataset_fps'), sample_size=sample_size)
        else:
            sim = max_similarity_to_dataset(smi, dataset, sample_size=sample_size)
    except Exception:
        sim = max_similarity_to_dataset(smi, dataset, sample_size=sample_size)
    if not RDKit_AVAILABLE:
        return float(sim)

    valid = 1.0 if is_valid_smiles(smi) else 0.0
    q = qed_score(smi) if valid else 0.0
    # weights
    w_valid = 0.4
    w_qed = 0.4
    w_sim = 0.2
    reward = w_valid * valid + w_qed * q + w_sim * sim
    # clamp
    return float(max(0.0, min(1.0, reward)))


def sequence_penalty(smi: str, vocab_size: int, alpha: float = 0.6, beta: float = 0.4) -> float:
    """Compute a penalty in [0,1] for simplistic/repetitive sequences.

    - entropy component: low character entropy increases penalty
    - run-length component: long runs of the same token increase penalty

    alpha and beta control the relative weighting. The function returns a
    combined penalty in [0,1].
    """
    if not smi:
        return 1.0
    # frequency-based entropy
    from math import log

    counts = {}
    for c in smi:
        counts[c] = counts.get(c, 0) + 1
    L = len(smi)
    probs = [v / L for v in counts.values()]
    # entropy in nats; normalize by log(vocab_size) if >1
    ent = -sum(p * (log(p) if p > 0 else 0.0) for p in probs)
    denom = log(vocab_size) if vocab_size > 1 else 1.0
    ent_norm = ent / denom if denom > 0 else 1.0
    ent_norm = max(0.0, min(1.0, ent_norm))
    penalty_entropy = 1.0 - ent_norm

    # run-length: longest run of identical chars relative to length
    max_run = 1
    cur = 1
    for i in range(1, L):
        if smi[i] == smi[i - 1]:
            cur += 1
            if cur > max_run:
                max_run = cur
        else:
            cur = 1
    run_rel = max_run / L
    # map run_rel in [0,1] to penalty roughly centered: small runs ok,
    # runs > 0.5 become penalized more heavily
    if run_rel <= 0.5:
        penalty_run = 0.0
    else:
        penalty_run = min(1.0, (run_rel - 0.5) * 2.0)

    penalty = alpha * penalty_entropy + beta * penalty_run
    penalty = max(0.0, min(1.0, penalty))
    return penalty


def _sample_from_logits(probs: np.ndarray) -> int:
    """Sample an index from probability vector (numpy)."""
    return int(np.random.choice(len(probs), p=probs))


if __name__ == "__main__":
    # Put heavy imports behind runtime guard so tests can import module without torch
    try:
        import torch
        import torch.nn as nn
        TORCH_AVAILABLE = True
    except ModuleNotFoundError:
        TORCH_AVAILABLE = False

    def verify_environment_and_exit(smiles):
        """Quick smoke test when torch isn't available: load data, build vocab,
        and compute rewards for a few examples to verify pipelines.
        """
        print("\nPyTorch not installed. Running verification-only mode (no training).\n")
        print(f"RDKit available: {RDKit_AVAILABLE}")
        if len(smiles) == 0:
            print("No SMILES loaded from dataset — cannot proceed with verification.")
            raise SystemExit(1)
        stoi, itos, max_len = build_vocab(smiles)
        print(f"Loaded {len(smiles)} unique SMILES, vocab_size={len(itos)}, max_len={max_len}")
        # sample a few SMILES from dataset and compute rewards
        sample = random.sample(smiles, min(10, len(smiles)))
        for s in sample:
            r = compute_reward(s, smiles, sample_size=200)
            print(f"SMI: {s[:120]}... reward={r:.4f} valid={is_valid_smiles(s) if RDKit_AVAILABLE else 'n/a'} qed={qed_score(s) if RDKit_AVAILABLE else 'n/a'}")
        print("\nVerification complete. Install PyTorch to run full training: `pip install torch` (or use conda).\n")
        raise SystemExit(0)

    # Small, easy-to-run configuration
    NUM_SMILES = 2000  # cap dataset to keep training quick
    EMBED_SIZE = 64
    HIDDEN_SIZE = 128
    BATCH_SIZE = 64
    PRETRAIN_EPOCHS = 3
    RL_STEPS = 300
    LR = 1e-3


    print("Loading dataset (this uses pyrfume)...")
    smiles = load_smiles(limit=NUM_SMILES)
    if len(smiles) == 0:
        raise SystemExit("No SMILES found in leffingwell dataset via pyrfume")

    # If RDKit is available, filter to only RDKit-parseable SMILES to avoid
    # noisy parser errors during training and to compute meaningful QED rewards.
    if RDKit_AVAILABLE:
        valid_smiles = [s for s in smiles if is_valid_smiles(s)]
        if len(valid_smiles) > 0:
            print(f"Filtered dataset: {len(valid_smiles)}/{len(smiles)} valid SMILES (using valid for training)")
            smiles = valid_smiles
        else:
            print("Warning: no valid SMILES found after RDKit filtering; using original dataset")

    # Build dataset fingerprints to speed up Tanimoto reward computation
    dataset_fps = None
    if RDKit_AVAILABLE:
        print("Building Morgan fingerprints for dataset (this may take a few seconds)...")
        dataset_fps = build_fingerprints(smiles)
        print(f"Built {len(dataset_fps)} fingerprints")

    if not TORCH_AVAILABLE:
        verify_environment_and_exit(smiles)

    stoi, itos, max_len = build_vocab(smiles)
    vocab_size = len(itos)
    print(f"Dataset size: {len(smiles)} SMILES, vocab size: {vocab_size}, max_len: {max_len}")

    # Encode dataset into integer sequences (with start/end tokens)
    def encode(s: str):
        return [stoi["^"]] + [stoi[c] for c in s] + [stoi["$"]]

    encoded = [encode(s) for s in smiles]

    # Pad sequences to max_len+2 maybe
    PAD = stoi["$"]

    def collate(batch):
        L = max(len(x) for x in batch)
        arr = np.full((len(batch), L), PAD, dtype=np.int64)
        for i, x in enumerate(batch):
            arr[i, : len(x)] = x
        return torch.from_numpy(arr)

    class SMILESModel(nn.Module):
        def __init__(self, vocab_size, embed_size, hidden_size):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, embed_size)
            self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, vocab_size)

        def forward(self, x, hidden=None):
            emb = self.embed(x)
            out, hidden = self.lstm(emb, hidden)
            logits = self.fc(out)
            return logits, hidden

        def sample(self, max_len=200, temperature=1.0, device='cpu', require_valid=False, max_attempts: int = 5):
            """Sample a SMILES string. If require_valid and RDKit is available,
            will try up to `max_attempts` attempts to generate a RDKit-parseable SMILES.
            """
            self.eval()
            with torch.no_grad():
                for attempt in range(max_attempts):
                    tokens = [stoi["^"]]
                    h = None
                    for _ in range(max_len):
                        x = torch.tensor([[tokens[-1]]], dtype=torch.long).to(device)
                        logits, h = self.forward(x, h)
                        probs = torch.softmax(logits[0, -1] / temperature, dim=-1).cpu().numpy()
                        idx = _sample_from_logits(probs)
                        tokens.append(idx)
                        if idx == stoi["$"]:
                            break
                    chars = [itos[i] for i in tokens[1:-1]]
                    smi = "".join(chars)
                    if require_valid and RDKit_AVAILABLE:
                        if is_valid_smiles(smi):
                            return smi
                        else:
                            # try again
                            continue
                    else:
                        return smi
                # if we get here, return last sampled (may be invalid)
                return smi

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SMILESModel(vocab_size, EMBED_SIZE, HIDDEN_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # Prepare simple data loader indices
    indices = list(range(len(encoded)))

    # Prepare replay buffer (used by checkpoint loader and RL loop)
    replay = ReplayBuffer(max_size=1000)

    # Try to load checkpoint if it exists
    ckpt_dir = "checkpoints"
    ckpt_path = os.path.join(ckpt_dir, "latest.pt")
    loaded_step = None
    try:
        loaded_step = load_checkpoint(ckpt_path, model, optimizer, replay)
        if loaded_step is not None:
            print(f"Loaded checkpoint from {ckpt_path} at step {loaded_step}")
    except Exception:
        pass

    # Pretraining (MLE)
    print("Starting supervised pretraining (MLE)...")
    for epoch in range(PRETRAIN_EPOCHS):
        random.shuffle(indices)
        t0 = time.time()
        total_loss = 0.0
        count = 0
        for i in range(0, len(indices), BATCH_SIZE):
            batch_idx = indices[i : i + BATCH_SIZE]
            batch = [encoded[j] for j in batch_idx]
            batch_tensor = collate(batch).to(device)
            inputs = batch_tensor[:, :-1]
            targets = batch_tensor[:, 1:]

            logits, _ = model(inputs)
            # flatten
            logits_flat = logits.reshape(-1, vocab_size)
            targets_flat = targets.reshape(-1)
            loss = criterion(logits_flat, targets_flat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            count += 1
        print(f"Epoch {epoch+1}/{PRETRAIN_EPOCHS} loss={total_loss/count:.4f} time={time.time()-t0:.1f}s")

    # Show some samples before RL
    print("Example molecules (before RL):")
    for _ in range(5):
        print(model.sample(max_len=max_len + 5, device=device))

    # Reinforcement learning via REINFORCE
    print("Starting REINFORCE fine-tuning...")
    PENALTY_WEIGHT = 0.8  # weight for multiplicative penalty on reward
    NOVELTY_WEIGHT = 0.2  # additive novelty reward weight
    for step in range(RL_STEPS):
        # Sample a batch of molecules
        model.train()
        batch_samples = []
        batch_logp = []
        batch_rewards = []
        batch_penalties = []
        # validity gating stats
        gating_rejections = 0
        gating_attempts = 0
        MAX_GATING_ATTEMPTS = 4
        for _ in range(BATCH_SIZE):
            # try up to MAX_GATING_ATTEMPTS to sample a valid SMILES when RDKit available
            sampled = None
            sampled_tokens = None
            sampled_logp = None
            for attempt in range(MAX_GATING_ATTEMPTS):
                gating_attempts += 1
                # sample sequence token-by-token while accumulating log probs
                tokens = [stoi["^"]]
                logp = 0.0
                h = None
                for _ in range(max_len + 5):
                    x = torch.tensor([[tokens[-1]]], dtype=torch.long).to(device)
                    logits, h = model(x, h)
                    probs = torch.softmax(logits[0, -1], dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    idx = dist.sample().item()
                    logp += dist.log_prob(torch.tensor(idx, device=device)).item()
                    tokens.append(idx)
                    if idx == stoi["$"]:
                        break
                smi = "".join([itos[i] for i in tokens[1:-1]])
                if RDKit_AVAILABLE:
                    if is_valid_smiles(smi):
                        sampled = smi
                        sampled_tokens = tokens
                        sampled_logp = logp
                        break
                    else:
                        gating_rejections += 1
                        continue
                else:
                    sampled = smi
                    sampled_tokens = tokens
                    sampled_logp = logp
                    break
            # fallback: take last sampled even if invalid
            if sampled is None:
                sampled = smi
                sampled_tokens = tokens
                sampled_logp = logp

            base_reward = compute_reward(sampled, smiles, sample_size=500)
            pen = sequence_penalty(sampled, vocab_size)
            reward_basic = float(base_reward * (1.0 - PENALTY_WEIGHT * pen))

            # novelty score relative to replay buffer (1 = novel)
            novelty = 0.0
            if RDKit_AVAILABLE and replay is not None and len(replay) > 0:
                novelty = compute_novelty_score(sampled, replay, sample_size=200)
            elif RDKit_AVAILABLE and len(replay) == 0:
                # if buffer empty, treat as maximally novel
                novelty = 1.0

            reward = float(max(0.0, min(1.0, reward_basic + NOVELTY_WEIGHT * novelty)))
            reward = max(0.0, min(1.0, reward))
            batch_samples.append((sampled_tokens, sampled))
            batch_logp.append(sampled_logp)
            batch_rewards.append(reward)
            batch_penalties.append(pen)
            # add valid samples to replay buffer
            if RDKit_AVAILABLE and is_valid_smiles(sampled):
                try:
                    fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(sampled), 2, nBits=dataset_fps[0].GetNumBits())
                    replay.add(sampled, fp)
                except Exception:
                    pass

        # baseline: average reward
        baseline = float(np.mean(batch_rewards))

        # compute policy gradient loss
        policy_loss = 0.0
        for lp, r in zip(batch_logp, batch_rewards):
            policy_loss += -(r - baseline) * lp
        policy_loss = policy_loss / BATCH_SIZE

        # convert to torch loss (scalar)
        optimizer.zero_grad()
        # create a dummy tensor to backprop through model parameters using autograd
        # we use autograd to compute grads by re-computing log probs; simpler approach
        # is to re-run forward and compute log probs with torch tensors, but for
        # brevity we approximate by using negative policy loss as a scalar and
        # calling backward on it with create_graph=False. To keep gradients correct
        # we recompute using torch tensors properly below.

        # Proper gradient computation: re-run sampled sequences and compute log probs (torch)
        total_loss = torch.tensor(0.0, device=device)
        for (tokens, _smi), r in zip(batch_samples, batch_rewards):
            h = None
            seq_logp = torch.tensor(0.0, device=device)
            for t in range(len(tokens) - 1):
                x = torch.tensor([[tokens[t]]], dtype=torch.long).to(device)
                logits, h = model(x, h)
                probs = torch.softmax(logits[0, -1], dim=-1)
                dist = torch.distributions.Categorical(probs)
                seq_logp = seq_logp + dist.log_prob(torch.tensor(tokens[t + 1], device=device))
            total_loss = total_loss + -(r - baseline) * seq_logp

        total_loss = total_loss / BATCH_SIZE
        total_loss.backward()
        optimizer.step()

        if (step + 1) % 50 == 0 or step == 0:
            avg_r = float(np.mean(batch_rewards))
            avg_pen = float(np.mean(batch_penalties)) if len(batch_penalties) > 0 else 0.0
            # detach total_loss for safe printing
            print(f"Step {step+1}/{RL_STEPS} avg_reward={avg_r:.4f} avg_penalty={avg_pen:.4f} novelty_buf={len(replay)} gating_rej={gating_rejections}/{gating_attempts} baseline={baseline:.4f} loss={float(total_loss.detach()):.6f}")

        # periodic checkpointing
        if (step + 1) % 100 == 0:
            try:
                save_checkpoint(ckpt_path, model, optimizer, replay, step + 1)
                print(f"Saved checkpoint to {ckpt_path} at step {step+1}")
            except Exception as e:
                print("Warning: failed to save checkpoint:", e)

    # Show some samples after RL
    print("Example molecules (after RL):")
    for _ in range(10):
        print(model.sample(max_len=max_len + 5, device=device))

    # final checkpoint
    try:
        save_checkpoint(ckpt_path, model, optimizer, replay, RL_STEPS)
        print(f"Saved final checkpoint to {ckpt_path}")
    except Exception as e:
        print("Warning: failed to save final checkpoint:", e)
