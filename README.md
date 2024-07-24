# 🤖 LLAM 
##Seamless AI Assistant API for OpenAI, Azure & Ollama models

Welcome to the **FastAPI-based AI Assistant Management API**! This project provides an interface to create, manage, and interact with AI assistants using various providers like OpenAI, Azure, and Ollama.

## 🛠️ Features

- **Providers Supported**: OpenAI, Azure, and Ollama.
- **Assistant Management**: Create, retrieve, update, and delete AI assistants.
- **Thread Management**: Create threads for conversational context and manage messages.
- **Run Management**: Execute tasks and processes using assistants, handling tool outputs and status updates.
- **In-Memory Storage**: Store assistants, threads, and runs in-memory for quick access.

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- FastAPI
- Uvicorn

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-repo/your-project.git
    cd your-project
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

To start the FastAPI server, run the following command:

```bash
uvicorn main:app --reload
