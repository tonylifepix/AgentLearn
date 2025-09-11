from init import init
import os
import json
import re
import pyrfume
import numpy as np  # Add if not already imported
from cerebras.cloud.sdk import Cerebras

from libserach import search_smile_by_description

def main():
    init()
    client = Cerebras(
        # This is the default and can be omitted
        api_key=os.environ.get("CEREBRAS_API_KEY"),
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_smile_by_description",
                "strict": True,
                "description": "A tool for searching SMILES by odor properties",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "The odor properties to search for, e.g., ['apple', 'woody']. The valid properties are ['alcoholic', 'aldehydic', 'alliaceous', 'almond', 'animal', 'anisic', 'apple', 'apricot', 'aromatic', 'balsamic', 'banana', 'beefy', 'berry', 'black currant', 'brandy', 'bread', 'brothy', 'burnt', 'buttery', 'cabbage', 'camphoreous', 'caramellic', 'catty', 'chamomile', 'cheesy', 'cherry', 'chicken', 'chocolate', 'cinnamon', 'citrus', 'cocoa', 'coconut', 'coffee', 'cognac', 'coumarinic', 'creamy', 'cucumber', 'dairy', 'dry', 'earthy', 'ethereal', 'fatty', 'fermented', 'fishy', 'floral', 'fresh', 'fruity', 'garlic', 'gasoline', 'grape', 'grapefruit', 'grassy', 'green', 'hay', 'hazelnut', 'herbal', 'honey', 'horseradish', 'jasmine', 'ketonic', 'leafy', 'leathery', 'lemon', 'malty', 'meaty', 'medicinal', 'melon', 'metallic', 'milky', 'mint', 'mushroom', 'musk', 'musty', 'nutty', 'odorless', 'oily', 'onion', 'orange', 'orris', 'peach', 'pear', 'phenolic', 'pine', 'pineapple', 'plum', 'popcorn', 'potato', 'pungent', 'radish', 'ripe', 'roasted', 'rose', 'rum', 'savory', 'sharp', 'smoky', 'solvent', 'sour', 'spicy', 'strawberry', 'sulfurous', 'sweet', 'tea', 'tobacco', 'tomato', 'tropical', 'vanilla', 'vegetable', 'violet', 'warm', 'waxy', 'winey', 'woody']."
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]

    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant with access to search engine specific for finding SMILES string with ODOR properties. Use the search_smile_by_description tool to find SMILES string of molecules with exact or similar odor when needed. If the user proposed properties are not within the valid set, imagine what in the valid properties combines or alone would smell similar to the requested one. The valid properties are ['alcoholic', 'aldehydic', 'alliaceous', 'almond', 'animal', 'anisic', 'apple', 'apricot', 'aromatic', 'balsamic', 'banana', 'beefy', 'berry', 'black currant', 'brandy', 'bread', 'brothy', 'burnt', 'buttery', 'cabbage', 'camphoreous', 'caramellic', 'catty', 'chamomile', 'cheesy', 'cherry', 'chicken', 'chocolate', 'cinnamon', 'citrus', 'cocoa', 'coconut', 'coffee', 'cognac', 'coumarinic', 'creamy', 'cucumber', 'dairy', 'dry', 'earthy', 'ethereal', 'fatty', 'fermented', 'fishy', 'floral', 'fresh', 'fruity', 'garlic', 'gasoline', 'grape', 'grapefruit', 'grassy', 'green', 'hay', 'hazelnut', 'herbal', 'honey', 'horseradish', 'jasmine', 'ketonic', 'leafy', 'leathery', 'lemon', 'malty', 'meaty', 'medicinal', 'melon', 'metallic', 'milky', 'mint', 'mushroom', 'musk', 'musty', 'nutty', 'odorless', 'oily', 'onion', 'orange', 'orris', 'peach', 'pear', 'phenolic', 'pine', 'pineapple', 'plum', 'popcorn', 'potato', 'pungent', 'radish', 'ripe', 'roasted', 'rose', 'rum', 'savory', 'sharp', 'smoky', 'solvent', 'sour', 'spicy', 'strawberry', 'sulfurous', 'sweet', 'tea', 'tobacco', 'tomato', 'tropical', 'vanilla', 'vegetable', 'violet', 'warm', 'waxy', 'winey', 'woody']. Always call the tool when the user is asking for SMILES with specific odor properties."},
        {
            "role": "user",
            "content": "What molecule smells like noctua fans?",
        }
    ]


    response = client.chat.completions.create(
        messages=messages,
        model="qwen-3-235b-a22b-thinking-2507",
        tools=tools,
        parallel_tool_calls=False,
    )
    
    choice = response.choices[0].message
    print("Model response:", choice)

    if choice.tool_calls:
        function_call = choice.tool_calls[0].function
        if function_call.name == "search_smile_by_description":
            # Logging that the model is executing a function named "calculate".
            print(f"Model executing function '{function_call.name}' with arguments {function_call.arguments}")

            # Parse the arguments from JSON format and perform the requested calculation.
            arguments = json.loads(function_call.arguments)
            result = search_smile_by_description(arguments["query"])

            # Note: This is the result of executing the model's request (the tool call), not the model's own output.
            print(f"Search result sent to model: {result}")

        # Send the result back to the model to fulfill the request.
            messages.append({
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": choice.tool_calls[0].id
            })
    
        # Request the final response from the model, now that it has the calculation result.
            final_response = client.chat.completions.create(
                model="qwen-3-235b-a22b-thinking-2507",
                messages=messages,
            )
            
            # Handle and display the model's final response.
            if final_response:
                print("Final model output:", final_response.choices[0].message.content)
            else:
                print("No final response received")
    else:
        # Handle cases where the model's response does not include expected tool calls.
        print("Unexpected response from the model")

if __name__ == "__main__":
    main()