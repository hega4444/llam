import uuid
import uvicorn
import json
import copy
import asyncio
from typing import Union, List, Optional, Dict, Type

from fastapi import FastAPI, HTTPException, BackgroundTasks, Body
from contextlib import asynccontextmanager

from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import StructuredTool, BaseTool
from langchain.pydantic_v1 import BaseModel, Field, create_model

from LLAM.tools import all_tools
from LLAM.connectors.ollama import OllamaConnector, OllamaCompletion
from LLAM.connectors.openai_client import OpenaiClient, to_openai_format, convert_to_openai_tool
from LLAM.connectors.azure import AzureOAIClient
from LLAM.config import LLAM_HOST, LLAM_PORT
from LLAM.common import *


# Define providers
providers = {
    "ollama": OllamaConnector,
    "openai": OpenaiClient(),
    "azure": AzureOAIClient(),
}

def minilog(message):
    filename = "minilog.log"
    with open(filename, "a") as file:
        file.write(message + "\n")

def none():
    return

# Define logic at startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup

    # Start the listen_to_redis function as a background task on startup
    app.process_pending_runs = asyncio.create_task(process_pending_runs())
        
    yield # Server running

    # Cancel the background task when the FastAPI application is shutting down
    app.process_pending_runs.cancel()
    await app.process_pending_runs

    print("Background tasks closed.")


# Define the main app
app = FastAPI(lifespan=lifespan)

# Classes to replace dicts
class Assistant:
    def __init__(self, id, provider, model, name, instructions, tools):
        self.id = id
        self.provider = provider
        self.model = model
        self.name = name
        self.instructions = instructions
        self.tools = tools

class Thread:
    def __init__(self, id, assistant):
        self.id = id
        self.assistant = assistant
        self.messages = []
        self.metadata = ""

class Run:
    def __init__(self, id, assistant, thread, instructions, tools, connector=None):
        self.id = id
        self.assistant = assistant
        self.thread = thread
        self.status = "queued"
        self.instructions = instructions
        self.tools = tools
        self.required_action = None
        self.connector = connector
        self.last_error = None
        self._llam_action = None
        self._llam_outputs = None

# In-memory storage
assistants = {}
threads = {}
runs = {}


# Process pending runs
async def process_pending_runs():

    while True:
        for run_id, run in runs.items():
            if run._llam_action == "process_outputs":

                asyncio.create_task(process_tool_outputs(run.assistant, run, run._llam_outputs))
            
                run._llam_action = None
                run._llam_outputs = None

        await asyncio.sleep(1)

# Background task function
def execute_run(run, connector, messages, submit : bool = False):
    try:
        minilog(f"Running connector Ollama - Model {connector.model} - Run {run.id}")
        run.status = "in_progress"  
        result_messages = connector.run(messages, run, submit)
        run.thread.messages = result_messages
    except Exception as e:
        run.status = "failed" 

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


def convert_to_langchain_tool(tool: dict) -> StructuredTool:

    try:
        lc_tool = StructuredTool.from_function(
            name=tool["function"]["name"],
            description=tool["function"]["description"],
            args_schema=dict_to_pydantic_model(name="args_schema", schema=tool["function"]["parameters"]),
            func= none
        )
        return lc_tool
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def chat_completion_create_backend(request: ChatCompletionRequest):

    if request.provider.lower() not in providers:
        raise HTTPException(status_code=404, detail="Provider not found")
    
    if request.provider.lower() in ("openai", "azure"):
        client = providers[request.provider.lower()]

        # Convert messages to OpenAI format
        messages_oai = [{"role": message.type, "content": message.content} for message in request.messages]

        try:
            response = client.chat.completions.create(
                model= request.model, 
                messages= messages_oai,
                temperature= request.temperature
            )

            return response.choices[0].message.content

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    if request.provider.lower() == "ollama":
        connector = OllamaCompletion(
            model=request.model,
            temperature=request.temperature,    
        )
        try:
            # Convert messages to Langchain format
            messages_lc = []
            for message in request.messages:
                if message.type.lower() == "user":
                    langchain_message = HumanMessage(content=message.content)
                elif message.type.lower() == "system":
                    langchain_message = SystemMessage(content=message.content)
                elif message.type.lower() == "assistant":
                    langchain_message = AIMessage(content=message.content)
                else:
                    raise HTTPException(status_code=400, detail="Invalid message type")

                messages_lc.append(langchain_message)

            response = connector.run(messages_lc)

            return response

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

async def process_tool_outputs(assistant, run, outputs):

    last_user_message = None
    for message in run.thread.messages:
        if message.type.lower() == "human":
            last_user_message = message.content

    completion_request = ChatCompletionRequest(
        provider= assistant.provider, 
        model= assistant.model, 
        messages= [
            SimpleMessage(type="user", content=last_user_message or "Requested tool:"),
            SimpleMessage(type="assistant", content=f"Previous function call outputs {outputs}"),
            SimpleMessage(type="system", content=f"Generate an answer for the user using these previous function call outputs.")
        ]
    )

    run.status = "in_progress"
    try:
        # Schedule the connector to run in the background
        response = await chat_completion_create_backend(completion_request)

        await append_messages_backend(
            thread_id= run.thread.id,
            messages= SimpleMessage(type="assistant", content=f"{response.content}"),
        )

        run.status = "completed"

    except Exception as e:
        run.status = "failed"


# Endpoints

@app.post("/chat_completion_create")
async def chat_completion_create(request: ChatCompletionRequest):

    response = await chat_completion_create_backend(request)

    return response


@app.post("/create_assistant")
async def create_assistant(request: CreateAssistantRequest):

    assistant_id = None

    if request.provider.lower() not in providers:
        raise HTTPException(status_code=404, detail="Provider not found")
    
    if request.provider.lower() in ("openai", "azure"):
        client = providers[request.provider.lower()]

        tools_oai = request.tools

        if request.tools and isinstance(request.tools[0], str):

            try:
                # Convert tools to OpenAI format
                tools_oai = [convert_to_openai_tool(all_tools[tool]) for tool in request.tools]
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        try:
            response = client.beta.assistants.create(
                model= request.model, 
                name= request.name, 
                instructions= request.instructions, 
                tools= tools_oai
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        assistant_id = response.id

    if request.tools and isinstance(request.tools[0], str):
        tools_lc = [all_tools[tool] for tool in request.tools]

    elif request.tools and isinstance(request.tools[0], dict):
        # Convert tools to Langchain format
        tools_lc = [convert_to_langchain_tool(tool) for tool in request.tools]

    
    if not assistant_id:
        assistant_id = str(uuid.uuid4())

    assistant = Assistant(
        id=assistant_id,
        provider=request.provider.lower(),
        model=request.model,
        name=request.name,
        instructions=request.instructions,
        tools=tools_lc
    )

    assistants[assistant_id] = assistant

    response = copy.copy(assistant)
    response.tools = [str(tool) for tool in assistant.tools]

    return vars(response)


@app.get("/retrieve_assistant/{assistant_id}")
async def retrieve_assistant(assistant_id: str):
    if assistant_id not in assistants:
        raise HTTPException(status_code=404, detail="Assistant not found")

    assistant = assistants[assistant_id]

    if assistant.provider in ("openai", "azure"):
        client = providers[assistant.provider]
        response = client.beta.assistants.retrieve(assistant_id)
        
        return response

    return vars(assistant)

@app.post("/update_assistant/{assistant_id}")
async def update_assistant(assistant_id: str, request: UpdateAssistantRequest):
    if assistant_id not in assistants:
        raise HTTPException(status_code=404, detail="Assistant not found")

    assistant = assistants[assistant_id]

    if request.name is not None:
        assistant.name = request.name
    if request.instructions is not None:
        assistant.instructions = request.instructions
    if request.tools is not None:
        assistant.tools = request.tools

    if assistant.provider in ("openai", "azure"):
        client = providers[assistant.provider]
        response = client.beta.assistants.update(assistant_id, **request.dict())
        
        return response

    return {"status": "Assistant updated successfully"}

@app.post("/create_thread/{assistant_id}")
async def create_thread(assistant_id: str):

    thread_id = None

    if assistant_id not in assistants:
        raise HTTPException(status_code=404, detail="Assistant not found")
    
    assistant = assistants[assistant_id]

    if assistant.provider in ("openai", "azure"):
        client = providers[assistant.provider]
        response = client.beta.threads.create()
        
        thread_id = response.id

    if not thread_id:
        thread_id = str(uuid.uuid4())
    
    if thread_id in threads:
        raise HTTPException(status_code=409, detail="Thread already exists")

    thread = Thread(id=thread_id, assistant=assistant)
    threads[thread_id] = thread

    response = copy.copy(thread)
    response.assistant = thread.assistant.id

    return vars(response)

async def append_messages_backend(thread_id: str, messages: Union[SimpleMessage, List[SimpleMessage]]):
    if thread_id not in threads:
        raise HTTPException(status_code=404, detail="Thread not found")

    thread = threads[thread_id]

    # Convert single message to list for unified processing
    if not isinstance(messages, list):
        messages = [messages]

    for message in messages:
        langchain_message = None

        if message.type.lower() == "user":
            langchain_message = HumanMessage(content=message.content)
        elif message.type.lower() == "system":
            langchain_message = SystemMessage(content=message.content)
        elif message.type.lower() == "assistant":
            langchain_message = AIMessage(content=message.content)
        else:
            raise HTTPException(status_code=400, detail="Invalid message type")

        thread.messages.append(langchain_message)

    return {"status": "Message(s) appended"}

@app.post("/append_messages/{thread_id}")
async def append_messages(thread_id: str, messages: Union[SimpleMessage, List[SimpleMessage]]):
    
    await append_messages_backend(thread_id, messages)

    thread = threads[thread_id]
    assistant = thread.assistant

    if assistant.provider in ("openai", "azure"):
        client = providers[assistant.provider]

        openai_messages = [to_openai_format(message) for message in messages]

        for message in openai_messages:
            response = client.beta.threads.messages.create(thread_id, **message)
        
        return {"status": "Message(s) updated"}

    return {"status": "Message(s) updated"}


@app.post("/update_metadata/{thread_id}")
async def update_metadata(thread_id: str, metadata: dict = Body(...)):
    if thread_id not in threads:
        raise HTTPException(status_code=404, detail="Thread not found")

    thread = threads[thread_id]
    thread.metadata = str(metadata)

    return {"status": "Metadata updated successfully"}

@app.get("/retrieve_messages/{thread_id}")
async def retrieve_messages(thread_id: str):
    if thread_id not in threads:
        raise HTTPException(status_code=404, detail="Thread not found")

    thread = threads[thread_id]
    assistant = thread.assistant
    simple_messages = []

    if assistant.provider in ("openai", "azure"):
        client = providers[assistant.provider]
        response = client.beta.threads.messages.list(thread_id=thread.id)
        
        for message in response.data:
            simple_messages.append(SimpleMessage(type=message.role, content=message.content[0].text.value))

        return {"messages": simple_messages}
    
    for langchain_message in thread.messages:
        message_type = {
            "HumanMessage": "user",
            "SystemMessage": "system",
            "AIMessage": "assistant",
        }[langchain_message.__class__.__name__]

        content = langchain_message.content

        simple_message = SimpleMessage(type=message_type, content=content)
        simple_messages.append(simple_message)
        simple_messages.reverse()

    return {"messages": simple_messages}

@app.post("/create_run/{thread_id}")
async def create_run(
    thread_id: str,
    background_tasks : BackgroundTasks,
    instructions: Optional[str] = None,
    tools: Optional[List[str]] = None,
    additional_messages: Optional[List[SimpleMessage]] = None,
):
   
    run_id = None
    connector = None
    if thread_id not in threads:
        raise HTTPException(status_code=404, detail="Thread not found")

    thread = threads[thread_id]

    # Append additional messages
    if additional_messages:
        await append_messages_backend(thread_id, additional_messages)

    # Check assistant provider
    assistant = thread.assistant

    # Update tools or use default tools
    tools = tools or assistant.tools

    if assistant.provider in ("openai", "azure"):
        client = providers[assistant.provider]
        connector = client

        try:
            if tools and isinstance(tools[0], str):
                # Convert tools to OpenAI format
                tools_oai = [convert_to_openai_tool(all_tools[tool]) for tool in tools]

            elif tools and isinstance(tools[0], StructuredTool):
                tools_oai = [convert_to_openai_tool(tool) for tool in tools]

            response = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant.id,
                instructions=instructions or assistant.instructions,
                tools=tools_oai,
            )

            run_id = response.id
        except Exception as e:
            raise HTTPException(status_code=500, detail=e)
    
    if not run_id:
        run_id = str(uuid.uuid4())

    run = Run(
        id=run_id,
        assistant=assistant,
        thread=thread,
        instructions=instructions,
        tools=tools,
        connector=connector
    )

    runs[run_id] = run

    if tools and isinstance(tools[0], str):
        # Convert tool references to langchain
        tools_lc = [all_tools[tool] for tool in tools or []]
    else:
        tools_lc = tools

    if assistant.provider == "ollama":
        # Call Ollama API
        connector = OllamaConnector(
            messages= thread.messages,
            model= assistant.model, 
            name= assistant.name, 
            instructions= instructions or assistant.instructions, 
            tools= tools_lc,
            format="json"
        )

        run.connector = connector
        minilog(f"Run {assistant.provider} {run_id} created")

        # Schedule the connector to run in the background
        background_tasks.add_task(execute_run, run, connector, thread.messages)

    
    run_formatted = copy.copy(run)
    run_formatted.connector = connector.__class__.__name__
    run_formatted.assistant = thread.assistant.id
    run_formatted.thread = thread.id
    run_formatted.tools = [str(tool) for tool in run_formatted.tools]

    return run_formatted
    

@app.get("/retrieve_run_status/{run_id}")
async def retrieve_run_status(run_id: str):
    if run_id not in runs:
        raise HTTPException(status_code=404, detail="Run not found")
    
    run = runs[run_id]
    assistant = run.assistant

    if assistant.provider in ("openai", "azure"):
        client = providers[assistant.provider]
        response = client.beta.threads.runs.retrieve(thread_id=run.thread.id, run_id=run_id)
    
        if response.required_action and \
            response.required_action.type == "submit_tool_outputs":

            tool_calls = response.required_action.submit_tool_outputs.tool_calls
            for tool in tool_calls:
                tool.function.arguments = json.loads(tool.function.arguments)

        return response

    run_formatted = copy.copy(run)
    run_formatted.connector = run.connector.__class__.__name__
    run_formatted.assistant = run.assistant.id
    run_formatted.thread = run.thread.id
    run_formatted.tools = [str(tool) for tool in run_formatted.tools]

    return run_formatted

@app.post("/cancel_run/{run_id}")
async def cancel_run(run_id: str):
    if run_id not in runs:
        raise HTTPException(status_code=404, detail="Run not found")
    runs[run_id].status = "cancelled"

    run = runs[run_id]
    assistant = run.assistant

    if assistant.provider in ("openai", "azure"):
        client = providers[assistant.provider]
        response = client.beta.threads.runs.cancel(thread_id=run.thread.id, run_id=run_id)
        
        return response

    return {"run_id": run_id, "status": "cancelled"}

@app.post("/submit_tool_outputs/{run_id}")
async def submit_tool_outputs(
    background_tasks: BackgroundTasks,
    run_id: str, 
    tool_outputs: List[ToolOutput]
):
    if run_id not in runs:
        raise HTTPException(status_code=404, detail="Run not found")

    run = runs[run_id]

    # Update the backend with the received tool outputs
    run.tool_outputs = tool_outputs
    run.status = "in_progress"

    # Update the thread with the received tool outputs
    for call in tool_outputs:
        await append_messages_backend(
            thread_id= run.thread.id,
            messages= SimpleMessage(type="system", content=f"Output for tool call {call.tool_call_id}: {call.output}")
        )
    
    assistant = run.assistant

    # Invoke model
    if assistant.provider in ("openai", "azure"):
        client = providers[assistant.provider]
        response = client.beta.threads.runs.submit_tool_outputs(thread_id=run.thread.id, run_id=run_id, tool_outputs=tool_outputs)
        return response

    # Langchain connectors
    run._llam_action = "process_outputs"
    run._llam_outputs = tool_outputs

    return {"status": "Tool outputs submitted successfully"}

server = None

def main():
    global server
    server = uvicorn.run(app, host=LLAM_HOST, port=LLAM_PORT)

# To run the server, use the command: uvicorn filename:app --reload
if __name__ == "__main__":
    main()
