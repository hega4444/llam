# common.py
from typing import Any, List, Optional, Union
from pydantic import BaseModel
from langchain.schema import HumanMessage, SystemMessage, AIMessage

class DotDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(e)

# Pydantic models
class CreateAssistantRequest(BaseModel):
    provider: str
    model: str
    name: str
    instructions: str
    tools: list

class UpdateAssistantRequest(BaseModel):
    name: Optional[str] = None
    instructions: Optional[str] = None
    tools: Optional[List[str]] = None

class SimpleMessage(BaseModel):
    type: str # ("user", "system", "assistant")
    content: str

class ToolOutput(BaseModel):
    tool_call_id: str
    output: Any

class ChatCompletionRequest(BaseModel):
    provider: str
    model: str
    messages: List[SimpleMessage]
    temperature: Optional[float] = None
    response_format: Optional[str] = None
