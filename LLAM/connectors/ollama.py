# test_langhain with function calling and LexiExternalCommand

from LLAM.connectors.ollama_functions import OllamaFunctions
from langchain_core.messages import SystemMessage

class OllamaConnector():

    def __init__(self, messages, model, name, instructions, tools=None, format = "json"):
        self.model = model
        self.name = name
        self.format = format
        self.instructions = instructions
        self.tools = tools
        self.model_instance = OllamaFunctions(model=model, format=format)
        self.status = "created"

        if self.name:
            messages.insert(0, (SystemMessage(f"You are {self.name}. Always use the name provided.")))
        if self.instructions:
            messages.insert(0, (SystemMessage(self.instructions)))

    def run(self, messages, run, submit: bool = False):

        self.status = "in_progress"

        if self.tools and not submit:
            model_w_tools = self.model_instance.bind_tools(tools=self.tools)
            ai_msg = model_w_tools.invoke(messages)
        else:
            ai_msg = self.model_instance.invoke(messages)

        messages.append(ai_msg)

        if ai_msg.tool_calls:
            self.status = run.status = "requires_action"

            run.required_action = {
                    "type": "submit_tool_outputs",
                    "submit_tool_outputs": {
                        "tool_calls" : []
                    }
                }

            for tool_call in ai_msg.tool_calls:

                call_dict = {
                    "id": tool_call["id"],
                    "type": "function",
                    "function": {
                        "name": tool_call["name"],
                        "arguments": tool_call["args"]
                    }
                }

                run.required_action["submit_tool_outputs"]["tool_calls"].append(call_dict)

        else:
            self.status = run.status = "completed"

        return messages
    
class OllamaCompletion():
    def __init__(self, model, format = "json", temperature = 0.5):
        self.model = model
        self.format = format
        self.model_instance = OllamaFunctions(model=model, format=format)

    def run(self, messages):
        
        i_msg = self.model_instance.invoke(messages)

        return i_msg



