from datetime import datetime
from fastmcp import FastMCP
import random
import secrets
import string

mcp = FastMCP("Time Server")

@mcp.tool()

def current_time():

    """Return the current date and time."""

    return datetime.now().strftime(
        "%d-%m-%Y %H:%M:%S"
    )



@mcp.tool()

def roll_dice():

    """Roll a six-sided dice."""

    return random.randint(1,6)



@mcp.tool()

def generate_password(length: int = 12):

    """
    Generate a secure password.
    """

    alphabet = (
        string.ascii_letters
        + string.digits
        + string.punctuation
    )

    return "".join(
        secrets.choice(alphabet)
        for _ in range(length)
    )

if __name__ == "__main__":
    print("Starting MCP server..")
    mcp.run()
