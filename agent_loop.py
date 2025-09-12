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


async def main():

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
            # chat_prompt = "What color are Zebras? \n\n"
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
