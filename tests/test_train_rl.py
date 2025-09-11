import pytest
import pyrfume
from train_rl import load_smiles, build_vocab, max_similarity_to_dataset, compute_reward
from train_rl import RDKit_AVAILABLE


def test_pyrfume_loads():
    mol = pyrfume.load_data("leffingwell/molecules.csv")
    beh = pyrfume.load_data("leffingwell/behavior.csv")
    assert mol is not None
    assert beh is not None


def test_helpers_run_quickly():
    s = load_smiles(limit=10)
    assert isinstance(s, list)
    stoi, itos, max_len = build_vocab(s)
    assert isinstance(stoi, dict)
    # compute similarity for a short SMILES
    if len(s) > 0:
        sim = max_similarity_to_dataset(s[0], s, sample_size=5)
        assert 0.0 <= sim <= 1.0
        # reward should be within [0,1]
        r = compute_reward(s[0], s, sample_size=5)
        assert 0.0 <= r <= 1.0
        if RDKit_AVAILABLE:
            # if RDKit available and mol is parseable, reward should be >= 0
            assert r >= 0.0
