import pyrfume
import numpy as np  # Add if not already imported
import pandas as pd

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
    leffingwell_mol = pyrfume.load_data("leffingwell/molecules.csv")
    leffingwell_behavior = pyrfume.load_data("leffingwell/behavior.csv")
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

        results.append({"IsomericSMILES": smi, "name": name})

    return results

def search_description_by_smiles(query, top_n=10):
    return None

if __name__ == "__main__":
    query = ["floral", "fruity"]
    smiles = search_smile_by_description(query)
    print("Top 10 SMILES for query", query, ":", smiles)