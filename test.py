# Import things that are needed generically
from langchain.pydantic_v1 import BaseModel, Field, create_model
from langchain.tools import BaseTool, StructuredTool, tool
from typing import Any, List, Dict, Type

function = {'type': 'function', 'function': {'description': ' Multiply two numbers. Keys/words related to this function: "a", "b".', 'name': 'multiplyL', 'parameters': {'type': 'object', 'properties': {'a': {'type': 'number', 'description': 'The first number'}, 'b': {'type': 'number', 'description': 'The second number'}}, 'required': ['a', 'b']}}}


class CalculatorInput(BaseModel):
    a: int = Field(description="first number")
    b: int = Field(description="second number")

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


type_mapping = {
    'number': float,
    'integer': int,
    'string': str,
    'boolean': bool,
    'object': dict,
    'array': list
}

def convert_type(json_type: Any) -> Any:
    if isinstance(json_type, str):
        return type_mapping.get(json_type, Any)
    elif isinstance(json_type, dict) and json_type.get('type') == 'array':
        items_type = convert_type(json_type.get('items', {}).get('type'))
        return List[items_type]
    return Any

def dict_to_pydantic_model(name: str, schema: Dict[str, Any]) -> Type[BaseModel]:
    fields = {}
    for key, value in schema.get('properties', {}).items():
        field_type = convert_type(value)
        if value['type'] == 'object':
            # Recursively create nested Pydantic models for nested objects
            nested_model = dict_to_pydantic_model(key.capitalize(), value)
            fields[key] = (nested_model, Field(..., description=value.get('description', '')))
        else:
            # Handle basic types
            fields[key] = (field_type, Field(..., description=value.get('description', '')))
    return create_model(name, **fields)


args_schema = dict_to_pydantic_model('CalculatorInput', function['function']['parameters'])


calculator = StructuredTool.from_function(
    func=multiply,
    name="Calculator",
    description="multiply numbers",
    args_schema=args_schema,
    return_direct=True,
    # coroutine= ... <- you can specify an async method if desired as well
)

