from mcp.server.fastmcp import FastMCP
import torch
import numpy as np
from rdkit import Chem
from loguru import logger
from predictor_model import PredictorModel
import pickle

mcp = FastMCP("Chem and BioInformatics MCP Server")


current_model_path = "smiles_model.pt"
model = PredictorModel(input_dim=128, hidden_dim=64, output_dim=113)
model.load_state_dict(torch.load(current_model_path))
model.eval()

# This is the vector representation of all the molecules available for training
current_remaining_training_data_latents = np.load("remaining_training_data_latents.npy")
current_remaining_training_data_properties = np.load(
    "remaining_training_data_properties.npy"
)


# molecules used for training. This is a list of SMILES strings used for training
# not the full dataset but just the subset that has been used.
with open("remaining_training_data_smiles.pkl", "rb") as f:
    current_training_data_smiles = pickle.load(f)


def reload_data():
    global model
    global current_remaining_training_data_latents
    global current_remaining_training_data_properties
    global current_training_data_smiles

    current_model_path = "smiles_model.pt"
    model = PredictorModel(input_dim=128, hidden_dim=64, output_dim=113)
    model.load_state_dict(torch.load(current_model_path))
    model.eval()

    # This is the vector representation of all the molecules available for training
    current_remaining_training_data_latents = np.load(
        "remaining_training_data_latents.npy"
    )
    current_remaining_training_data_properties = np.load(
        "remaining_training_data_properties.npy"
    )

    # molecules used for training. This is a list of SMILES strings used for training
    # not the full dataset but just the subset that has been used.
    with open("remaining_training_data_smiles.pkl", "rb") as f:
        current_training_data_smiles = pickle.load(f)
    logger.info("Data reloaded successfully.")


def smiles_to_latent(smiles: str) -> np.ndarray:
    """
    Convert a SMILES string to its latent vector representation.
    """
    latent_vector = model.latent_representation(smiles)
    return latent_vector


@mcp.tool()
def most_unique_training_molecule(smiles: str) -> str:
    """
    Given a SMILES string, return the most unique molecule from the training data.
    Uniqueness is determined based on the latent space representation.
    """
    reload_data()
    logger.info(f"Finding most unique molecule for SMILES: {smiles}")
    # Convert SMILES to latent representation
    X = np.array([smiles_to_latent(smiles) for smiles in current_training_data_smiles])
    N = X.shape[0]
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    sims = X_norm @ X_norm.T
    dists = 1 - sims
    avg_dists = np.sum(dists, axis=1) - np.diag(dists)
    avg_dists = avg_dists / (N - 1)
    most_unique_index = np.argmax(avg_dists)
    return current_training_data_smiles[most_unique_index]


@mcp.tool()
def canonicalize_smiles(smiles: str) -> str:
    """
    Canonicalize a SMILES string.
    """
    try:
        logger.info(f"Canonicalizing SMILES: {smiles}")
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol)
    except Exception as e:
        return smiles


@mcp.tool()
def verify_smiles(smiles: str) -> bool:
    """
    Verify if a SMILES string is valid.
    """
    try:
        logger.info(f"Verifying SMILES: {smiles}")
        Chem.MolFromSmiles(smiles)
        return True
    except Exception as e:
        return False


@mcp.tool()
def is_known_smiles(smiles: str) -> str:
    """
    Check if a SMILES string is known (exists in the current training data).
    """
    try:
        Chem.MolFromSmiles(smiles)  # Validate SMILES
    except Exception as e:
        return "Invalid SMILES"
    return "Known" if smiles in current_training_data_smiles else "New"


@mcp.tool()
def get_mol_from_pool(smiles: str, top_k: int = 5) -> list:
    """
    Given a SMILES string, return the top_k most similar molecules from the available data.

    Similarity is determined based on the latent space representation.
    """
    reload_data()
    try:
        logger.info(f"Getting similar molecules for SMILES: {smiles}")
        # Convert SMILES to latent representation
        # Here we assume a function `smiles_to_latent` exists that converts SMILES to latent vector
        latent_vector = smiles_to_latent(smiles)  # This function needs to be defined

        # Calculate distances to all training data in latent space
        distances = np.linalg.norm(
            current_remaining_training_data_latents - latent_vector, axis=1
        )

        # Get indices of top_k closest molecules
        top_k_indices = np.argsort(distances)[:top_k]

        # Retrieve corresponding SMILES strings
        similar_smiles = [current_training_data_smiles[i] for i in top_k_indices]
        logger.info(f"Similar molecules found: {similar_smiles}")

        return similar_smiles
    except Exception as e:
        logger.error(f"Error in get_similar_molecules: {e}")
        return []


if __name__ == "__main__":
    mcp.run(transport="stdio")
