import pyrfume
import numpy as np  # Add if not already imported
import pandas as pd
import warnings

try:
    from rdkit import Chem
    RDKit_AVAILABLE = True
except Exception:
    RDKit_AVAILABLE = False

# Module-level caches to avoid repeated I/O / canonicalization
_LEFFINGWELL_MOL = None
_LEFFINGWELL_BEHAVIOR = None
_SMI_TO_CIDS_RAW = None
_SMI_TO_CIDS_CANON = None
_LEFFINGWELL_FPS = None


def _load_data_cached():
    """Load leffingwell datasets and cache them at module level."""
    global _LEFFINGWELL_MOL, _LEFFINGWELL_BEHAVIOR
    if _LEFFINGWELL_MOL is None or _LEFFINGWELL_BEHAVIOR is None:
        _LEFFINGWELL_MOL = pyrfume.load_data("leffingwell/molecules.csv")
        _LEFFINGWELL_BEHAVIOR = pyrfume.load_data("leffingwell/behavior.csv")
    return _LEFFINGWELL_MOL, _LEFFINGWELL_BEHAVIOR


def _build_smiles_maps_cached(mol_df):
    """Return (raw_map, canonical_map). Caches results on first build.

    raw_map: mapping from dataset IsomericSMILES string -> [CIDs]
    canonical_map: same mapping but keys are RDKit-canonicalized SMILES when possible
    """
    global _SMI_TO_CIDS_RAW, _SMI_TO_CIDS_CANON
    if _SMI_TO_CIDS_RAW is not None and _SMI_TO_CIDS_CANON is not None:
        return _SMI_TO_CIDS_RAW, _SMI_TO_CIDS_CANON

    smi_series = mol_df.get("IsomericSMILES")
    raw_map = {}
    canon_map = {}
    if smi_series is None:
        _SMI_TO_CIDS_RAW, _SMI_TO_CIDS_CANON = raw_map, canon_map
        return raw_map, canon_map

    for cid, smi in smi_series.dropna().items():
        raw_map.setdefault(smi, []).append(cid)
        if RDKit_AVAILABLE:
            canon = canonicalize_smiles(smi)
            key = canon if canon is not None else smi
            canon_map.setdefault(key, []).append(cid)
        else:
            # when RDKit not available the canonical map should just mirror raw_map
            canon_map.setdefault(smi, []).append(cid)

    _SMI_TO_CIDS_RAW, _SMI_TO_CIDS_CANON = raw_map, canon_map
    return _SMI_TO_CIDS_RAW, _SMI_TO_CIDS_CANON


def canonicalize_smiles(smi):
    """Return RDKit-canonicalized SMILES or None if cannot be parsed or RDKit unavailable."""
    if smi is None:
        return None
    if not RDKit_AVAILABLE:
        return None
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return None


def _build_fingerprints_cached(mol_df, radius=2, nBits=2048):
    """Return a mapping CID -> RDKit fingerprint (cached).

    If RDKit is unavailable this returns an empty dict.
    """
    global _LEFFINGWELL_FPS
    if _LEFFINGWELL_FPS is not None:
        return _LEFFINGWELL_FPS

    fps = {}
    if not RDKit_AVAILABLE:
        _LEFFINGWELL_FPS = fps
        return fps

    try:
        from rdkit.Chem import AllChem
        from rdkit.DataStructs import ExplicitBitVect
    except Exception:
        _LEFFINGWELL_FPS = fps
        return fps

    smi_series = mol_df.get("IsomericSMILES")
    if smi_series is None:
        _LEFFINGWELL_FPS = fps
        return fps

    for cid, smi in smi_series.dropna().items():
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
            fps[cid] = fp
        except Exception:
            # skip molecules we can't parse / fingerprint
            continue

    _LEFFINGWELL_FPS = fps
    return fps


def evaluate_smiles_novelty(smi, top_n=5, similarity_threshold=0.8):
    """Evaluate how novel a SMILES string is relative to the Leffingwell dataset.

    Returns a dict with the contract:
      - query: original SMILES string
      - canonical_smiles: canonicalized SMILES (or None)
      - is_in_dataset: True if an exact (or canonical) match exists in dataset
      - matched_cids: list of dataset CIDs that match the query exactly
      - novel: boolean (True if not present and not similar above threshold)
      - max_similarity: highest Tanimoto similarity (float) or None if not computed
      - nearest: list of up to top_n nearest dataset entries as dicts {cid, similarity, IsomericSMILES, name}

    Behavior when RDKit is unavailable:
      - canonicalization and fingerprint-based similarity are not performed.
      - novelty reduces to whether the exact SMILES string appears in the dataset.
    """
    leffingwell_mol, _ = _load_data_cached()

    result = {
        "query": smi,
        "canonical_smiles": None,
        "is_in_dataset": False,
        "matched_cids": [],
        "novel": True,
        "max_similarity": None,
        "nearest": [],
    }

    if smi is None:
        return result

    # canonicalize when possible
    canon = canonicalize_smiles(smi)
    result["canonical_smiles"] = canon

    # Find exact / canonical matches using the cached maps
    smi_to_cids_raw, smi_to_cids_canon = _build_smiles_maps_cached(leffingwell_mol)
    smi_map = smi_to_cids_canon if RDKit_AVAILABLE else smi_to_cids_raw
    key = canon if (RDKit_AVAILABLE and canon is not None) else smi
    matched = list(smi_map.get(key, []))
    result["matched_cids"] = matched
    result["is_in_dataset"] = len(matched) > 0

    if not RDKit_AVAILABLE:
        # Can't compute similarity; novelty is simply presence/absence
        result["novel"] = not result["is_in_dataset"]
        return result

    # RDKit available: compute fingerprint similarity to the dataset
    try:
        from rdkit.Chem import AllChem
        from rdkit.DataStructs import TanimotoSimilarity
    except Exception:
        result["novel"] = not result["is_in_dataset"]
        return result

    # Build query fingerprint
    try:
        qmol = Chem.MolFromSmiles(canon if canon is not None else smi)
        if qmol is None:
            # unparsable SMILES
            result["novel"] = not result["is_in_dataset"]
            return result
        qfp = AllChem.GetMorganFingerprintAsBitVect(qmol, 2, nBits=2048)
    except Exception:
        result["novel"] = not result["is_in_dataset"]
        return result

    fps = _build_fingerprints_cached(leffingwell_mol)
    if not fps:
        result["novel"] = not result["is_in_dataset"]
        return result

    # Compute similarities
    sims = []
    for cid, fp in fps.items():
        try:
            sim = float(TanimotoSimilarity(qfp, fp))
        except Exception:
            continue
        sims.append((cid, sim))

    if len(sims) == 0:
        result["novel"] = not result["is_in_dataset"]
        return result

    # sort by similarity descending
    sims.sort(key=lambda x: x[1], reverse=True)
    result["max_similarity"] = float(sims[0][1])

    nearest = []
    for cid, sim in sims[:top_n]:
        row = None
        if cid in leffingwell_mol.index:
            mol_row = leffingwell_mol.loc[cid]
            if isinstance(mol_row, pd.DataFrame):
                mol_row = mol_row.iloc[0]
            smi_val = mol_row.get("IsomericSMILES") if hasattr(mol_row, 'get') else mol_row["IsomericSMILES"]
            name = None
            if hasattr(mol_row, 'index') and "name" in mol_row.index:
                name = mol_row.get("name")
            try:
                if pd.isna(name):
                    name = None
            except Exception:
                pass
        else:
            smi_val = None
            name = None

        nearest.append({"cid": cid, "similarity": float(sim), "IsomericSMILES": smi_val, "name": name})

    result["nearest"] = nearest

    # Novel if no exact match and highest similarity less than threshold
    result["novel"] = (not result["is_in_dataset"]) and (result["max_similarity"] < similarity_threshold)

    return result

# leffingwell_mol.head()
#                MolecularWeight             IsomericSMILES IUPACName                       name
# CID                                                                                           
# -955348933095          240.387      CCCCC=COC(=O)CCCCCCCC       NaN          Hexenyl nonanoate
# -923209957509          196.290   CC(=O)OCC1C=CC(C(C)C)CC1       NaN  Tetrahydrocuminyl acetate
# -874408321546          244.331  CCCCCCCCC(OC(C)=O)C(=O)OC       NaN    Methyl acetoxydecanoate
# -873963935677          198.306       CCCCC=COC(=O)C(C)CCC       NaN     Hexenyl methylvalerate
# -862841422647          148.271                CCCC(S)COCC       NaN    Ethoxymethylbutanethiol
# >>> leffingwell_behavior.head()
#                alcoholic  aldehydic  alliaceous  almond  animal  anisic  apple  ...  vanilla  vegetable  violet  warm  waxy  winey  woody
# Stimulus                                                                        ...                                                      
# -955348933095          0          0           0       0       0       0      0  ...        0          0       0     0     1      0      0
# -923209957509          0          0           0       0       0       0      0  ...        0          0       0     0     0      0      1
# -874408321546          0          0           0       0       0       0      0  ...        0          0       0     0     0      0      0
# -873963935677          0          0           0       0       0       0      1  ...        0          0       0     0     0      0      0
# -862841422647          0          0           0       0       0       0      0  ...        0          0       0     0     0      0      0

# [5 rows x 113 columns]

def search_smile_by_description(query, top_n=10):
    leffingwell_mol, leffingwell_behavior = _load_data_cached()
    """Return top_n matches as dicts containing IsomericSMILES and name.

    query may be a single descriptor string or an iterable/list of descriptor strings.
    Matches are measured using the Jaccard similarity between the binary descriptor
    vector for each stimulus and the binary query vector.
    """
    # Normalize query into a set of descriptor names
    if query is None:
        return []
    if isinstance(query, str):
        query = [query]
    try:
        query_set = set(query)
    except TypeError:
        # not iterable
        return []

    if len(query_set) == 0:
        return []

    # Use all behavior columns (these are the descriptors)
    properties = list(leffingwell_behavior.columns)

    # Build binary mask for the query in the same column order
    mask = np.array([1 if prop in query_set else 0 for prop in properties], dtype=int)

    # Get behavior data as binary matrix (rows correspond to stimuli / CIDs)
    binary_data = (leffingwell_behavior.values.astype(bool)).astype(int)

    # Vectorized Jaccard computation
    intersection = np.sum(np.logical_and(binary_data, mask), axis=1)
    union = np.sum(np.logical_or(binary_data, mask), axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        similarities = np.where(union > 0, intersection / union, 0.0)

    # Get indices sorted by similarity descending
    sorted_idx = np.argsort(similarities)[::-1]

    # Map indices back to CIDs (behavior DataFrame index)
    cids = leffingwell_behavior.index.values

    results = []
    for idx in sorted_idx[:top_n]:
        cid = cids[idx]
        # Look up the molecule row by CID in leffingwell_mol (CID is the index there)
        if cid not in leffingwell_mol.index:
            continue
        row = leffingwell_mol.loc[cid]
        # If multiple rows (duplicate index), take the first
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        smi = row.get("IsomericSMILES") if hasattr(row, 'get') else row["IsomericSMILES"]
        name = None
        if "name" in row.index if hasattr(row, 'index') else False:
            name = row.get("name")
        # Coerce pandas NA to None
        try:
            if pd.isna(name):
                name = None
        except Exception:
            pass

        results.append({"IsomericSMILES": smi, "name": name, "similarity": float(similarities[idx])})

    return results

def search_description_by_smiles(query, top_n=10):
    leffingwell_mol, leffingwell_behavior = _load_data_cached()

    def canonicalize_smiles(smi):
        """Return RDKit-canonicalized SMILES or None if cannot be parsed or RDKit unavailable."""
        if smi is None:
            return None
        if not RDKit_AVAILABLE:
            return None
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return None
            # Use isomeric SMILES to preserve stereochemistry when present
            return Chem.MolToSmiles(mol, isomericSmiles=True)
        except Exception:
            return None

    # Normalize query into an iterable of SMILES
    if query is None:
        return []
    if isinstance(query, str):
        query = [query]
    try:
        query_set = set(query)
    except TypeError:
        return []

    if len(query_set) == 0:
        return []

    # Build mapping from SMILES (or canonical SMILES when RDKit available)
    # -> list of CIDs (index of molecules df)
    smi_series = leffingwell_mol.get("IsomericSMILES")
    if smi_series is None:
        return []

    # Use cached mappings when available
    smi_to_cids_raw, smi_to_cids_canon = _build_smiles_maps_cached(leffingwell_mol)
    smi_to_cids = smi_to_cids_canon if RDKit_AVAILABLE else smi_to_cids_raw

    # Collect seed CIDs for the provided SMILES. If RDKit is available,
    # canonicalize the query SMILES before matching against the dataset map.
    seed_cids = []
    for smi in query_set:
        key = smi
        if RDKit_AVAILABLE:
            canon = canonicalize_smiles(smi)
            if canon is not None:
                key = canon
        if key in smi_to_cids:
            seed_cids.extend(smi_to_cids[key])

    if len(seed_cids) == 0:
        # no matching SMILES in dataset
        return []

    # Use behavior properties columns
    properties = list(leffingwell_behavior.columns)
    # Binary matrix for all stimuli
    binary_data = (leffingwell_behavior.values.astype(bool)).astype(int)

    # Build a combined mask (logical OR) of the seed CIDs' behavior vectors
    seed_mask = np.zeros(len(properties), dtype=int)
    for cid in seed_cids:
        if cid not in leffingwell_behavior.index:
            continue
        row = leffingwell_behavior.loc[cid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        try:
            vec = (row.values.astype(bool)).astype(int)
        except Exception:
            # If row is not numeric-like, build by column lookup
            vec = np.array([1 if row.get(p, 0) else 0 for p in properties], dtype=int)
        seed_mask = np.logical_or(seed_mask, vec)
    seed_mask = seed_mask.astype(int)

    # Vectorized Jaccard similarity between seed_mask and all rows
    intersection = np.sum(np.logical_and(binary_data, seed_mask), axis=1)
    union = np.sum(np.logical_or(binary_data, seed_mask), axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        similarities = np.where(union > 0, intersection / union, 0.0)

    # Sort by similarity descending
    sorted_idx = np.argsort(similarities)[::-1]
    cids = leffingwell_behavior.index.values

    results = []
    for idx in sorted_idx:
        cid = cids[idx]
        # skip the seed molecules themselves
        if cid in seed_cids:
            continue

        # ensure molecule information exists
        if cid not in leffingwell_mol.index:
            continue

        mol_row = leffingwell_mol.loc[cid]
        if isinstance(mol_row, pd.DataFrame):
            mol_row = mol_row.iloc[0]

        smi = mol_row.get("IsomericSMILES") if hasattr(mol_row, 'get') else mol_row["IsomericSMILES"]
        name = None
        if hasattr(mol_row, 'index') and "name" in mol_row.index:
            name = mol_row.get("name")
        try:
            if pd.isna(name):
                name = None
        except Exception:
            pass

        # get odor properties for this CID as a dict of property->0/1
        prop_dict = {}
        if cid in leffingwell_behavior.index:
            prop_row = leffingwell_behavior.loc[cid]
            if isinstance(prop_row, pd.DataFrame):
                prop_row = prop_row.iloc[0]
            for p in properties:
                try:
                    val = prop_row.get(p) if hasattr(prop_row, 'get') else prop_row[p]
                    prop_dict[p] = int(bool(val))
                except Exception:
                    prop_dict[p] = 0
        else:
            for p in properties:
                prop_dict[p] = 0

        results.append({"IsomericSMILES": smi, "name": name, "properties": prop_dict, "similarity": float(similarities[idx])})
        if len(results) >= top_n:
            break

    return results

if __name__ == "__main__":
    query = ["floral", "fruity"]
    smiles = search_smile_by_description(query)
    print("Top 10 SMILES for query", query, ":", smiles)
    query_smi = "CCOC1=CC=CC=C1"
    desc = search_description_by_smiles(query_smi)
    print("Top 10 descriptions for SMILES", query_smi, ":", desc)