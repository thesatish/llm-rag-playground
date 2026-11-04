import asyncio

from openai import OpenAI
from dotenv import load_dotenv
import os

# Load configuration
load_dotenv()

# Create AI client
# client = OpenAI(
#     base_url=os.getenv("BASE_URL"),
#     api_key=os.getenv("API_KEY")
# )

client = OpenAI(
    base_url=os.getenv("GEMINI_BASE_URL", os.getenv("BASE_URL")),
    api_key=os.getenv("API_KEY_G", os.getenv("API_KEY"))
)

from mcp_client import (
    connect,
    disconnect,
    discover_tools,
    execute_tool
)

# Build Tool Descriptions
def build_tool_descriptions(tools):

    descriptions = ""
    for tool in tools:
        descriptions += f"""
Tool:
{tool.name}

Description:
{tool.description}
----------------------------
"""

    return descriptions


def planner(user_request, tools):

    tool_descriptions = build_tool_descriptions(
        tools
    )

    ###Then create a prompt by combining user input and Tool Description.  and Give this prompt to Ollama. ###

    prompt = f"""
You are an AI Planner.

Available Tools

{tool_descriptions}

Instructions

1. Select the best tool.
2. Reply ONLY with the tool name.
3. Do not explain.

User Request

{user_request}
"""
# Give this prompt to ollama
    response = client.chat.completions.create (

        # model= os.getenv("MODEL"),
        model = os.getenv("GEMINI_MODEL", os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")),
        
        messages=[

            {
                "role": "user",
                "content": prompt
            }

        ]

    )

#Ollama returns the tool name
    tool_name = response.choices[0].message.content

    return tool_name.strip()


async def main():
# Step 1: Connect to the MCP Server.
    client = await connect()

    # Step 2: Discover all available tools.
    tools = await discover_tools(
        client
    )
    print()
    print("Available Tools")
    print("----------------")

    for tool in tools:
        print(tool.name)
    print() 

    ### Step 3: Planner: Take the User Input (Question). Create Tool Descriptions for each tool. Then create a prompt by combining user input and Tool Description.  and Give this prompt to Ollama. ###

    user_request = input(
        "User : "
    )

    tool_name = planner(
        user_request,
        tools
    )

    print()

    print(
        "Planner Selected:",
        tool_name
    )


    await disconnect(
        client
    )

if __name__ == "__main__":

    asyncio.run(main())




