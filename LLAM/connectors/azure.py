# azure_client.py

# Defines the AureOpenAI client to be used along the framework
# chg_lexi_003 Migrate to AzureOpenAI

import os

from openai import AzureOpenAI

# Retrieve system variables

try:
    api_key = os.environ.get("AOAI_API_KEY")
    azure_endpoint = os.environ.get("AOAI_API_BASE")
    api_version = os.environ.get("AOAI_API_VERSION")

except Exception as e:
    print(e)
    print("Cannot start the service. AzureOpenAI System Variables are not defined. Try 'nano ~./bashrc' ")
    exit()

def AzureOAIClient():

    return AzureOpenAI(
        api_key = api_key,
        azure_endpoint = azure_endpoint,
        api_version = api_version,
        )
