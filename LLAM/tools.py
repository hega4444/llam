# tools available for all models

from langchain_core.tools import tool

@tool
def multiply(a: float, b: float) -> float:
    """
    # summ: Multiply two numbers
    # keys: a, b
    # a 'description': "The first number"
    # b 'description': "The second number"
    """
    return a * b


all_tools = {"multiply": multiply}