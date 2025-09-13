from google import genai
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio
import os
from contextlib import AsyncExitStack
import pickle
import numpy as np

from train_model import _training_loop, retrain_model
from dataset import SmilesDataset
import time

client_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=client_key)

server_params = StdioServerParameters(
    command="/home/szaman5/AgentLearn/aglearn/bin/python3",
    args=["mcp_server.py"],
    env=None,
)


async def main(run_iteration=1):

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            system_prompt = (
                "You are a helpful SMILES generator "
                + "that can generate new molecules in a SMILES format and optimize"
                + " for novelty to train a model."
            )
            print("System prompt:", system_prompt)

            print("Prompt: \n")
            chat_prompt = (
                "Generate 1 new SMILES strings"
                + " for molecules similar to the most unique molecule."
                + " You can use the tools to find the most unique molecule."
                + " For each molecules you suggest "
                + " verify the SMILES,"
                + " check if it already known/"
                + " Get similar molecules from the sample pool."
                + " If a molecule is known or doesn't fit the criteria, move on and"
                + " generate a different one and try again."
                + " Once you have found a new molecule, get similar molecules from the pool."
                + " Return the list of molecules from the pool"
                + " in the format smiles1, smiles2, smiles3, smiles4, smiles5 \n\n"
            )
            print(chat_prompt)
            prompt = chat_prompt

            await session.initialize()

            response = await client.aio.models.generate_content(
                model="gemini-2.5-flash-preview-05-20",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    tools=[session],
                    system_instruction=system_prompt,
                    thinking_config=types.ThinkingConfig(
                        include_thoughts=True, thinking_budget=8192
                    ),
                ),
            )
            assert response is not None
            assert response.usage_metadata is not None
            print("Thoughts tokens:", response.usage_metadata.thoughts_token_count)
            print("\n\n")
            print("Response:\n", response.text)

            print("\n\n")
            new_mols = []
            assert response.candidates is not None
            assert len(response.candidates) > 0
            assert response.candidates[0] is not None
            assert response.candidates[0].content is not None
            assert response.candidates[0].content.parts is not None
            for part in response.candidates[0].content.parts:
                if not part.text:
                    continue
                if part.thought:
                    print("Thought:", part.text)
                else:
                    print("Text:", part.text)
                    try:
                        smiles_list = part.text.strip().split(",")
                        new_mols.extend(smiles_list)
                    except Exception as e:
                        print("Error:", e)
            print("Response text:", response.text)
            print("Response usage metadata:", response.usage_metadata)
            print("New molecules:", new_mols)
            print(
                "New molecules list:",
                [mol.strip() for mol in new_mols],
            )
            print("New molecules count:", len(new_mols))
            print("New molecules unique count:", len(set(new_mols)))
            print("New molecules unique list:", set(new_mols))
            return new_mols


if __name__ == "__main__":
    current_training_data = []
    # initialize with random set of molecules from training data
    with open("training_data_pool_full.pkl", "rb") as f:
        full_data = pickle.load(f)
    # print(f"Full data size: {full_data}")
    num_starting_mols = 200
    full_training_properties = np.load("training_data_properties.npy")
    current_training_data = full_data[:num_starting_mols]
    current_training_properties = full_training_properties[:num_starting_mols, :]

    smiles_to_prop_dict = {
        smi: prop for smi, prop in zip(full_data, full_training_properties)
    }

    score, model = retrain_model(
        current_training_data, current_training_properties, fname="smiles_model.pt"
    )

    remaining_data = full_data[num_starting_mols:]
    remaining_properties = full_training_properties[num_starting_mols:, :]

    remaining_latents = []
    for smi in remaining_data:
        latent = model.latent_representation(smi)
        remaining_latents.append(latent)
    remaining_latents = np.array(remaining_latents, dtype=np.float32)
    np.save("remaining_training_data_latents.npy", remaining_latents)
    np.save("remaining_training_data_properties.npy", remaining_properties)

    with open("remaining_training_data_smiles.pkl", "wb") as f:
        pickle.dump(remaining_data, f)

    scores = [score]
    for i in range(10):
        # Obtained new molecules from the agent
        new_mol_to_use = asyncio.run(main(run_iteration=i))

        # For testing, just use some random molecules from the pool
        # new_mol_to_use = full_data[
        #     num_starting_mols + i * 5 : num_starting_mols + (i + 1) * 5
        # ]
        print(f"New molecules to use for retraining: {new_mol_to_use}")

        # Get properties for the new molecules
        new_prop_list = []
        filtered_new_mols = []
        for smi in new_mol_to_use:
            if smi in smiles_to_prop_dict:
                if smi in remaining_data:
                    new_prop_list.append(smiles_to_prop_dict[smi])
                    remaining_data.remove(smi)
                    filtered_new_mols.append(smi)
        print(f"Filtered new molecules: {filtered_new_mols}")
        if len(new_prop_list) < 1:
            continue
        new_prop_list = np.array(new_prop_list)

        current_training_data.extend(filtered_new_mols)
        current_training_properties = np.vstack(
            [current_training_properties, new_prop_list]
        )
        if len(new_prop_list) == 0:
            print("No new valid molecules found, skipping retraining.")
            continue
        print(f"New properties shape: {current_training_properties.shape}")

        # Retrain the model with the new molecules
        score, model = retrain_model(
            current_training_data, current_training_properties, fname="smiles_model.pt"
        )
        scores.append(score)

        remaining_latents = []
        remaining_properties = []
        for smi in remaining_data:
            latent = model.latent_representation(smi)
            remaining_latents.append(latent)
            remaining_properties.append(smiles_to_prop_dict[smi])
        remaining_properties = np.array(remaining_properties, dtype=np.float32)
        np.save("remaining_training_data_properties.npy", remaining_properties)
        remaining_latents = np.array(remaining_latents, dtype=np.float32)
        np.save("remaining_training_data_latents.npy", remaining_latents)

        with open("remaining_training_data_smiles.pkl", "wb") as f:
            pickle.dump(remaining_data, f)

        time.sleep(30)

    print(f"Scores over iterations: {scores}")
