from google import genai
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio
import os
from contextlib import AsyncExitStack


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
                "Generate 5 new SMILES strings"
                + " for molecules similar to the most unique molecule."
                + " You can use the tools to find the most unique molecule."
                + " For each molecules you suggest "
                + " verify the SMILES,"
                + " check if it already known/"
                + " If not known, get similar molecules from the sample pool."
                + " If a molecule is known or doesn't fit the criteria, move on and"
                + " generate a different one and try again."
                + " The final output should just be a list of the unique molecules"
                + " in the format [smiles1, smiles2, smiles3, smiles4, smiles5] \n\n"
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
                    print("Thought:", part.thought.text)
                if part.tool_call:
                    print("Tool call:", part.tool_call)
                if part.tool_response:
                    print("Tool response:", part.tool_response)
                if part.text:
                    print("Text:", part.text)
                    new_mols.append(part.text)
            print("Response text:", response.text)
            print("Response usage metadata:", response.usage_metadata)
            print("New molecules:", new_mols)
            print(
                "New molecules list:",
                [mol.strip() for mol in new_mols if mol.strip().startswith("C")],
            )
            print("New molecules count:", len(new_mols))
            print("New molecules unique count:", len(set(new_mols)))
            print("New molecules unique list:", set(new_mols))
            return new_mols


if __name__ == "__main__":
    current_training_data = []

    for i in range(10):
        # Obtained new molecules from the agent
        new_mol_to_use = asyncio.run(main(run_iteration=i))

        # Retrain the model with the new molecules
        # retrain_model(new_mol_to_use)  # This function needs to be defined

        # Calculate and save model performance metrics
        # metrics = evaluate_model(model, validation_data)  # This function needs to be defined

        # Save the model
        # torch.save(model.state_dict(), f"smiles_model.pt")

        # save the new latents and smiles to the correct files
