import pyrfume
import numpy as np
from rdkit import Chem
import torch
import pandas as pd
from torch.utils.data import Dataset


def smiles_to_vec(smiles, vector_size=2048):
    """Convert SMILES string to a numerical vector using RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    fp = Chem.RDKFingerprint(mol, fpSize=vector_size)
    arr = np.zeros((1,), dtype=np.int8)
    arr.resize(fp.GetNumBits())
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32)


class SmilesDataset(Dataset):
    def __init__(self, smiles_list, properties):
        self.smiles_list = smiles_list
        self.vectors = [smiles_to_vec(smiles) for smiles in smiles_list]
        self.properties = torch.from_numpy(properties)

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        return self.vectors[idx], self.properties[idx]


if __name__ == "__main__":
    # Example usage
    smiles = "CC(=O)O[C@H](C)CCN"
    vector = smiles_to_vec(smiles)
    print(f"SMILES: {smiles}")
    print(f"Vector: {vector}")
    print(f"Vector type: {type(vector)}")
    print(f"Vector length: {len(vector)}")
    print(f"Vector sum (number of bits set): {np.sum(vector)}")

    data = pyrfume.load_data("leffingwell/molecules.csv")
    behavior_data = pyrfume.load_data("leffingwell/behavior.csv")
    tags = behavior_data.columns.unique().to_numpy()
    tag_to_index = {tag: i for i, tag in enumerate(tags)}
    print(tags, tag_to_index)
    # data = data.merge(behavior_data, on="MoleculeID")
    # print(behavior_data.columns)
    # smiles_list = data["IsomericSMILES"].tolist()
    # print(data[:10])
    print(f"Number of molecules: {len(data)} {len(behavior_data)}")
    new_data = pd.concat([data, behavior_data], axis=1)
    print(new_data)
    smiles_list = new_data["IsomericSMILES"].tolist()
    properties = np.zeros((len(new_data), len(tags)), dtype=np.float32)

    prop_df = new_data[tags].to_numpy()
    properties[:, :] = prop_df[:, :]
    print(f"Properties array shape: {properties.shape}")
    print(properties[:10].sum(axis=1))
    dataset = SmilesDataset(smiles_list, properties)
    print(f"Dataset size: {len(dataset)}")
    vec, prop = dataset[0]
    print(f"First vector: {vec}")
    print(f"First properties: {prop}")

    # for tag in tags:
    #     if row.get(tag, 0) == 1:
    #         print(f"Setting property for molecule {i}, tag {tag} {row.get(tag, 0)}")
    #     properties[i, tag_to_index[tag]] = 1.0
    rand_perm = np.random.permutation(np.arange(len(dataset)))
    smiles_list = [smiles_list[i] for i in rand_perm]
    properties = properties[rand_perm, :]
    print(f"SMILES list length: {len(smiles_list)}")

    print(f"Properties shape: {properties.shape}")
    split_idx = int(0.8 * len(dataset))
    train_dataset = SmilesDataset(smiles_list[:split_idx], properties[:split_idx, :])
    test_dataset = SmilesDataset(smiles_list[split_idx:], properties[split_idx:, :])

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    torch.save(train_dataset, "train_dataset.pt")
    torch.save(test_dataset, "test_dataset.pt")

    full_training_smiles = train_dataset.smiles_list
    print(f"Full training SMILES count: {len(full_training_smiles)}")
