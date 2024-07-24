# llam_client.py

import httpx
from typing import Optional, Union, Literal
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from pydantic import BaseModel

from LLAM.config import LLAM_HOST, LLAM_PORT

class DotDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(e)

class SimpleMessage(BaseModel):
    type: str
    content: str

class LLAMClient:

    class BadRequestError(Exception):
        pass

    def __init__(self, base_url, timeout=60):
        self.base_url = base_url
        self.timeout = timeout
        self.client = httpx.Client(timeout=self.timeout)
        self.assistants = self.Assistant(self)
        self.threads = self.Thread(self)
        self.files = self.File(self)
        self.chat = self.Chat(self)
    
    class Chat:
        def __init__(self, client):
            self.client = client
            self.completions = self.Completions(self.client)
        
        class Completions:
            def __init__(self, client):
                self.client = client

            def create(self, provider, model, messages, temperature= 0.5, response_format={ "type": "json_object" }, input_format="langchain"):
                
                if not isinstance(messages, list) and messages is not None:
                    messages = [messages]

                simple_messages = []
                for message in messages:
                    if input_format == "openai":
                        simple_messages.append({
                            "type": message["role"],
                            "content": message["content"]
                        })

                    elif input_format == "langchain":
                        simple_messages.append({
                            "type": message.type if message.type != "human" else "user",
                            "content": message.content
                        })

                response = self.client.client.post(f"{self.client.base_url}/chat_completion_create", json={
                    "provider": provider,
                    "model": model,
                    "messages": simple_messages,
                    "temperature": temperature,
                    "response_format": str(response_format)
                })  

                response.raise_for_status()
                
                return response.json()

    class Assistant:
        def __init__(self, client):
            self.client = client
            self.files = self.File(self)

        def create(self, provider, model, name, instructions, tools):

            response = self.client.client.post(f"{self.client.base_url}/create_assistant", json={
                "provider": provider,
                "model": model,
                "name": name,
                "instructions": instructions,
                "tools": tools
            })
            
            response.raise_for_status()
            return DotDict(response.json())
        
        def retrieve(self, assistant_id):
            response = self.client.client.get(f"{self.client.base_url}/retrieve_assistant/{assistant_id}")
            response.raise_for_status()
            return DotDict(response.json())
        
        def update(self, assistant_id, name=None, instructions=None, tools=None):
            payload = {}
            if name is not None:
                payload["name"] = name
            if instructions is not None:
                payload["instructions"] = instructions
            if tools is not None:
                payload["tools"] = tools

            response = self.client.client.post(f"{self.client.base_url}/update_assistant/{assistant_id}", json=payload)
            response.raise_for_status()
            return DotDict(response.json())
    
        class File:
            def __init__(self, assistant):
                self.assistant = assistant
                self.client = assistant.client

            def create(self, assistant_id, file_id):
                # TODO: Implement file upload
                pass

    class Thread:
        def __init__(self, client):
            self.client = client
            self.runs = self.Run(self)
            self.messages = self.Messages(self)

        def create(self, assistant_id, messages=None, metadata=None):
            response = self.client.client.post(f"{self.client.base_url}/create_thread/{assistant_id}")
            response.raise_for_status()

            response = response.json()

            if messages or metadata:
                self.messages.create(response["id"], messages, metadata)

            return DotDict(response)

        class Run:
            def __init__(self, thread):
                self.thread = thread
                self.client = thread.client

            def create(
                    self, 
                    thread_id, 
                    instructions: Optional[str] = None, 
                    tools: Optional[list] = None, 
                    additional_messages: Optional[list] = None, 
                    assistant_id: Optional[str] = None,
                    model: Optional[str] = None,
            ):
                
                payload = {"instructions": instructions, "tools": tools, "additional_messages": additional_messages}
                response = self.client.client.post(f"{self.client.base_url}/create_run/{thread_id}", json=payload)
                response.raise_for_status()
                response = response.json()
                return DotDict(response)

            def retrieve(self, run_id, thread_id = None):
                response = self.client.client.get(f"{self.client.base_url}/retrieve_run_status/{run_id}")
                response.raise_for_status()
                return DotDict(response.json())

            def cancel(self, run_id):
                response = self.client.client.post(f"{self.client.base_url}/cancel_run/{run_id}")
                response.raise_for_status()
                return response.json()["status"]
            
            def submit_tool_outputs(self, run_id, tool_outputs, thread_id = None):
                response = self.client.client.post(f"{self.client.base_url}/submit_tool_outputs/{run_id}", json=tool_outputs)
                response.raise_for_status()
                return response.json()["status"]
        
        class Messages:
            def __init__(self, thread):
                self.thread = thread
                self.client = thread.client

            def create(self, thread_id, messages, metadata=None):
                # Convert a single message to a list for unified processing
                if not isinstance(messages, list) and messages is not None:
                    messages = [messages]

                converted_messages = []

                if messages:

                    # Convert each langchain message to SimpleMessage
                    for message in messages:
                        # Extracting content from the langchain message
                        content = DotDict(message).content

                        # Converting langchain message type to a simplified type
                        if isinstance(message, HumanMessage):
                            message_type = "user"
                        elif isinstance(message, SystemMessage):
                            message_type = "system"
                        elif isinstance(message, AIMessage):
                            message_type = "assistant"
                        elif isinstance(message, dict) and "role" in message:
                            message_type = message["role"]
                        else:
                            raise ValueError("Invalid message type")

                        # Creating a simplified SimpleMessage object
                        simple_message = SimpleMessage(type=message_type, content=content)
                        converted_messages.append(simple_message)

                        # Sending the simplified message(s) to the server
                        response = self.client.client.post(f"{self.client.base_url}/append_messages/{thread_id}", json=[msg.dict() for msg in converted_messages])
                        response.raise_for_status()

                # Update the metadata
                if metadata is not None:
                    data = {"metadata": str(metadata)}
                    response = self.client.client.post(f"{self.client.base_url}/update_metadata/{thread_id}", json=data)
                    response.raise_for_status()

                return response.json()["status"]

            def list(self, thread_id: str, format: Literal["langchain", "openai"] = "langchain") -> Union[list[dict], list]:
                response = self.client.client.get(f"{self.client.base_url}/retrieve_messages/{thread_id}")
                response.raise_for_status()

                if format == "openai":
                    # If format is 'openai', return the messages as received
                    messages = response.json()["messages"]
                    messages = [DotDict(msg) for msg in messages]
                    return messages

                # Default to 'langchain' format
                converted_messages = []

                for message_dict in response.json()["messages"]:
                    message_type = message_dict["type"]
                    content = message_dict["content"]

                    if format == "openai":
                        formatted_message = {"role": message_type, "content": content}

                    elif message_type == "user":
                        formatted_message = HumanMessage(content=content)
                    elif message_type == "system":
                        formatted_message = SystemMessage(content=content)
                    elif message_type == "assistant":
                        formatted_message = AIMessage(content=content)
                    else:
                        raise ValueError(f"Invalid message type: {message_type}")
                    
                    converted_messages.append(formatted_message)
                    
                return converted_messages
        
    class File:
        def __init__(self, client):
            self.client = client

        def create(self, file, purpose):
            # TODO: Implement file upload
            pass

if __name__ == "__main__":
    base_url = f"http://{LLAM_HOST}:{LLAM_PORT}"
    client = LLAMClient(base_url, timeout=300)

    # Example usage:
    provider = "ollama"
    model = "llama3:70b"
    name = "IB Assistant"
    instructions = "Help the user with their queries."
    tools = ["search", "translate", "summarize"]
    tools = []

    # Create an assistant
    assistant_id = client.assistants.create(provider, model, name, instructions, tools)
    print(f"Created assistant with ID: {assistant_id}")

    # Update assistant
    updated_assistant = client.assistants.update(assistant_id, name="New Name", instructions="New Instructions", tools=[])
    print(f"Updated Assistant: {updated_assistant}")

    # Retrieve assistant
    assistant = client.assistants.retrieve(assistant_id)
    print(f"Retrieve Assistant: {assistant}")

    # Create a thread
    thread_id = client.threads.create(assistant_id, metadata={"key": "value"})
    print(f"Created thread with ID: {thread_id}")

    # Append messages to the thread
    message = HumanMessage("Hello, how can I help you?")
    append_status = client.threads.messages.create(thread_id, message)
    print(f"Appended message to thread: {append_status}")

    # Retrieve messages from the thread
    messages = client.threads.messages.list(thread_id)
    print(f"Messages in thread: {messages}")

    # Create a run
    run_id = client.threads.runs.create(thread_id)
    print(f"Created run with ID: {run_id}")

    # Retrieve run status
    status = client.threads.runs.retrieve(run_id)
    print(f"Run status: {status}")

    # Cancel the run
    cancel_status = client.threads.runs.cancel(run_id)
    print(f"Cancelled run: {cancel_status}")



