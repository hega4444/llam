# openai_connector.py


from typing import Any
from openai import Client
from langchain.schema import HumanMessage, SystemMessage, AIMessage

from LLAM.common import SimpleMessage


def OpenaiClient():
    return
    return Client()

def to_openai_format(message: SimpleMessage) -> dict:

    openai_message = {
        "role": message.type,
        "content": message.content,
    }

    return openai_message

def convert_to_openai_tool(tool: Any) -> dict:
    """Convert a tool to an Ollama tool."""

    try:
        schema = tool.__dict__["args_schema"].schema()
        definition = {
            "type": "function",
            "function": {
                "description": tool.description,
                "name": tool.name,
                "parameters": schema,
            }
        }

        if "required" in schema:
            definition["function"]["parameters"]["required"] = schema["required"]

        return definition
    except Exception as e:
        raise ValueError(
            f"Cannot convert {tool} to an OpenAI tool. {e}."
        )