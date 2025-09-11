import importlib
import sys
import types
import pandas as pd


def _make_fake_pyrfume():
    mod = types.SimpleNamespace()

    def load_data(name):
        if name == "leffingwell/molecules.csv":
            # simple molecules DataFrame with CIDs 1 and 2
            df = pd.DataFrame({
                "IsomericSMILES": ["CCO", "CCC"],
                "name": ["ethanol_like", "propane_like"],
            }, index=[1, 2])
            df.index.name = "CID"
            return df
        if name == "leffingwell/behavior.csv":
            # behavior with two properties
            df = pd.DataFrame({
                "floral": [0, 1],
                "fruity": [1, 0],
            }, index=[1, 2])
            return df
        raise KeyError(name)

    mod.load_data = load_data
    return mod


def _import_with_fake_pyrfume():
    # inject fake pyrfume before importing module
    sys.modules["pyrfume"] = _make_fake_pyrfume()
    # ensure a fresh import
    if "libserach" in sys.modules:
        del sys.modules["libserach"]
    return importlib.import_module("libserach")


def test_evaluate_smiles_novelty_in_dataset():
    libserach = _import_with_fake_pyrfume()
    res = libserach.evaluate_smiles_novelty("CCO")
    assert res["is_in_dataset"] is True
    assert res["novel"] is False


def test_evaluate_smiles_novelty_out_of_dataset():
    libserach = _import_with_fake_pyrfume()
    res = libserach.evaluate_smiles_novelty("CCCCCC")
    # not an exact match in the fake dataset
    assert res["is_in_dataset"] is False
    # RDKit likely not available in test environment, so novel should be True
    assert res["novel"] is True
