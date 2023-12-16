# Langri-La
Langrila is a useful tool to use ChatGPT with OpenAI API or Azure in an easy way. This library put emphasis on simple architecture for readability.

# Dependencies
## must
- Python >=3.10,<3.13
- openai
- tiktoken

## as needed
- chroma or qdrant-client (for retrieval)

# Installation
## clone
```
git clone git@github.com:taikinman/langrila.git
```

## pip
```
cd langrila
pip install -e .
```

## poetry
```
poetry add --editable /path/to/langrila
```

# Usage example
## Pre-requirement
1. Pre-configure environment variables to use OpenAI API or Azure OpenAI Service.

## Basic usage
### For OpenAI
```python
from langrila import OpenAIChatModule

# For conventional model
chat = OpenAIChatModule(
    api_key_name = "API_KEY", # env variable name
    model_name="gpt-3.5-turbo-16k-0613",
    # organization_id_name="ORGANIZATION_ID", # env variable name
    timeout=60, 
    max_retries=2, 
)

message = "Please give me only one advice to improve the quality of my sleep."
response = chat(message)

# # for asynchronous processing
# response = await chat(message, arun=True)

response.model_dump()

>>> {'message': {'role': 'assistant',
  'content': 'Establish a consistent sleep schedule by going to bed and waking up at the same time every day, even on weekends.'},
 'usage': {'prompt_tokens': 21, 'completion_tokens': 23},
 'prompt': [{'role': 'user',
   'content': 'Please give me only one advice to improve the quality of my sleep.'}]}
```

For newest model released on 2023/11/6.
```python
chat = OpenAIChatModule(
    api_key_name = "API_KEY", # env variable name
    model_name="gpt-4-1106-preview", # you can specify newest model
    timeout=60, 
    max_retries=2, 
    # seed=42, # as needed
    # response_format={"type":"json_object"} # as needed
)
```

### For Azure
```python
chat = OpenAIChatModule(
        api_key_name="AZURE_API_KEY", # env variable name
        model_name="gpt-3.5-turbo-16k-0613",
        api_type="azure",
        api_version="2023-07-01-preview", 
        deployment_id_name="DEPLOY_ID", # env variable name
        endpoint_name="ENDPOINT", # env variable name
    )

```

## Batch processing
```python
messages = [
    "Please give me only one advice to improve the quality of my sleep.", 
    "Please give me only one advice to improve my memory.",
    "Please give me only one advice on how to make exercise a habit.",
    "Please give me only one advice to help me not get bored with things so quickly." 
            ]

await chat.abatch_run(messages, batch_size=4)

>>> [CompletionResults(message={'role': 'assistant', 'content': 'Establish a consistent sleep schedule by going to bed and waking up at the same time every day, even on weekends.'}, usage=Usage(prompt_tokens=21, completion_tokens=23, total_tokens=44), prompt=[{'role': 'user', 'content': 'Please give me only one advice to improve the quality of my sleep.'}]),
 CompletionResults(message={'role': 'assistant', 'content': 'One advice to improve memory is to practice regular physical exercise. Exercise has been shown to enhance memory and cognitive function by increasing blood flow to the brain and promoting the growth of new brain cells. Aim for at least 30 minutes of moderate-intensity exercise, such as brisk walking or jogging, most days of the week.'}, usage=Usage(prompt_tokens=18, completion_tokens=64, total_tokens=82), prompt=[{'role': 'user', 'content': 'Please give me only one advice to improve my memory.'}]),
 CompletionResults(message={'role': 'assistant', 'content': "Start small and be consistent. Start with just a few minutes of exercise each day and gradually increase the duration and intensity over time. Consistency is key, so make it a priority to exercise at the same time every day or on specific days of the week. By starting small and being consistent, you'll be more likely to stick with it and make exercise a long-term habit."}, usage=Usage(prompt_tokens=21, completion_tokens=76, total_tokens=97), prompt=[{'role': 'user', 'content': 'Please give me only one advice on how to make exercise a habit.'}]),
 CompletionResults(message={'role': 'assistant', 'content': 'One advice to help you not get bored with things so quickly is to cultivate a sense of curiosity and explore new perspectives. Instead of approaching tasks or activities with a fixed mindset, try to approach them with an open mind and a desire to learn something new. Embrace the mindset of a beginner and seek out different ways to engage with the task at hand. By continuously seeking novelty and finding new angles to approach things, you can keep your interest alive and prevent boredom from setting in.'}, usage=Usage(prompt_tokens=24, completion_tokens=96, total_tokens=120), prompt=[{'role': 'user', 'content': 'Please give me only one advice to help me not get bored with things so quickly.'}])]
```

## Stream
```python

prompt = "Please give me only one advice to improve the quality of my sleep."
response = chat(prompt, stream=True)

for c in response:
    # print(c, end="\r") # for flush
    print(c)

# # For async process
# response = chat(prompt, stream=True, arun=True)

# async for c in response:
#     print(c)

>>> Establish
Establish a
Establish a consistent
Establish a consistent sleep
Establish a consistent sleep schedule
Establish a consistent sleep schedule by
Establish a consistent sleep schedule by going
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Establish a consistent sleep schedule by going to bed and waking up at the same time every day, even on weekends.
message={'role': 'assistant', 'content': 'Establish a consistent sleep schedule by going to bed and waking up at the same time every day, even on weekends.'} usage=Usage(prompt_tokens=21, completion_tokens=30, total_tokens=51) prompt=[{'role': 'user', 'content': 'Please give me only one advice to improve the quality of my sleep.'}] # return the CompletionResults instance at the end
```

## Function Calling
### For OpenAI
```python
from langrila import ToolConfig, ToolParameter, ToolProperty, OpenAIFunctionCallingModule

def get_weather(city: str, date: str) -> str:
    return f"The weather in {city} on {date} is sunny."

tool_config = ToolConfig(
            name="get_weather",
            description="Get weather at current location.",
            parameters=ToolParameter(
                            properties=[
                                ToolProperty(
                                    name="city",
                                    type="string",
                                    description="City name"
                                ),
                                ToolProperty(
                                    name="date",
                                    type="string",
                                    description="Date"
                                )
                            ],
                            required=["city", "date"]
                        )     
                    )  

                    
client = OpenAIFunctionCallingModule(
        api_key_name="API_KEY", 
        model_name = "gpt-3.5-turbo-1106",
        tools=[get_weather], 
        tool_configs=[tool_config],
        seed=42,
    )


response = await client("Please tell me the weather in Tokyo on 2023/12/13", arun=True)
response.model_dump()

>>> {'usage': {'prompt_tokens': 71, 'completion_tokens': 24},
 'results': [{'call_id': 'call_NRCqCcPhbRWygXq26h7Pll9n',
   'funcname': 'get_weather',
   'args': '{"city":"Tokyo","date":"2023/12/13"}',
   'output': 'The weather in Tokyo on 2023/12/13 is sunny.'}],
 'prompt': [{'role': 'user',
   'content': 'Please tell me the weather in Tokyo on 2023/12/13'}]}

```

### For Azure
```python
formatter = OpenAIFunctionCallingModule(
            api_key_name = "AZURE_API_KEY", # env variable name
            model_name="gpt-3.5-turbo-16k-0613",
            api_type="azure", 
            api_version="2023-07-01-preview",
            endpoint_name="ENDPOINT", # env variable name
            deployment_id_name="DEPLOY_ID", # env variable name
            timeout=60,
            max_retries=2,
            tools=[get_weather], 
            tool_configs=[tool_config],
        )
```

## Conversation memory

```python
from langrila import JSONConversationMemory

chat = OpenAIChatModule(
    api_key_name = "API_KEY", # env variable name
    model_name="gpt-3.5-turbo-16k-0613",
    conversation_memory=JSONConversationMemory("./conversation_memory.json"),
    timeout=60, 
    max_retries=2, 
)


message = "Do you know Rude who is the character in Final Fantasy 7."
response = chat(message)
response.model_dump()
>>> {'message': {'role': 'assistant',
  'content': 'Yes, I am familiar with Rude, who is a character in the popular video game Final Fantasy 7. Rude is a member of the Turks, an elite group of operatives working for the Shinra Electric Power Company. He is known for his bald head, sunglasses, and calm demeanor. Rude often partners with another Turk named Reno, and together they carry out various missions throughout the game.'},
 'usage': {'prompt_tokens': 22, 'completion_tokens': 81},
 'prompt': [{'role': 'user',
   'content': 'Do you know Rude who is the character in Final Fantasy 7.'}]}


message = "What does he think about Tifa?"
response = chat(message)
response.model_dump()
>>> {'message': {'role': 'assistant',
  'content': "Rude's thoughts and feelings about Tifa, another character in Final Fantasy 7, are not explicitly stated in the game. However, it is known that Rude respects Tifa as a formidable fighter and acknowledges her skills. In the game, Rude and Tifa have a few encounters where they engage in combat, but there is no indication of any personal feelings or opinions Rude may have towards Tifa beyond their professional interactions."},
 'usage': {'prompt_tokens': 119, 'completion_tokens': 88},
 'prompt': [{'role': 'user',
   'content': 'Do you know Rude who is the character in Final Fantasy 7.'},
  {'role': 'assistant',
   'content': 'Yes, I am familiar with Rude, who is a character in the popular video game Final Fantasy 7. Rude is a member of the Turks, an elite group of operatives working for the Shinra Electric Power Company. He is known for his bald head, sunglasses, and calm demeanor. Rude often partners with another Turk named Reno, and together they carry out various missions throughout the game.'},
  {'role': 'user', 'content': 'What does he think about Tifa?'}]}
```

## Conversation memory with Cosmos DB

```python
from langrila.memory.cosmos import CosmosConversationMemory

chat = OpenAIChatModule(
    api_key_name = "API_KEY", # env variable name
    model_name="gpt-3.5-turbo-16k-0613",
    conversation_memory=CosmosConversationMemory(
        endpoint_name = "COSMOS_ENDPOINT", # env variable names
        key_name = "COSMOS_KEY", 
        dbname_name = "COSMOS_DB_NAME", 
        containername_name = "COSMOS_CONTAINER_NAME", 
        pkey_name = "COSMOS_PKEY"
        ),
    api_type="azure",
    api_version="2023-07-01-preview", 
    deployment_id_name="DEPLOY_ID", # env variable name
    endpoint_name="ENDPOINT", # env variable name
)
```

## Assembling module for specific use case
### Using prompt template

```python
from langrila import BaseModule, OpenAIChatModule, PromptTemplate
from typing import Optional
        
SAMPLE_TEMPLATE = """Please answer Yes or No to the following questions.

[Question]
{question}

[Answer]
"""

class TemplateChat(BaseModule):
    def __init__(
        self,
        api_key_name: str,
        model_name: str,
        max_tokens: int = 2048,
        timeout: int = 60,
        max_retries: int = 2,
        context_length: Optional[int] = None,
    ):
        self.chat = OpenAIChatModule(
            api_key_name=api_key_name,
            model_name=model_name,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            context_length=context_length,
        )


    def run(self, prompt: str, prompt_template: Optional[PromptTemplate] = None):
        if isinstance(prompt_template, PromptTemplate):
            prompt = prompt_template.set_args(question=prompt).format()
        response = self.chat(prompt)
        return response


prompt_template = PromptTemplate(
    template=SAMPLE_TEMPLATE,
)

chat = TemplateChat(
    api_key_name="API_KEY", 
    model_name = "gpt-3.5-turbo-1106",
)

prompt = "Are you GPT-4?"
response = chat(prompt, prompt_template=prompt_template)
response.model_dump()

>>> {'message': {'role': 'assistant', 'content': 'No'},
 'usage': {'prompt_tokens': 30, 'completion_tokens': 1},
 'prompt': [{'role': 'user',
   'content': 'Please answer Yes or No to the following questions.\n\n[Question]\nAre you GPT-4?\n\n[Answer]\n'}]}
```

### Retrieval
Now only Chroma and Qdrant are supported for basic retrieval.

#### For Chroma
```python
from langrila import OpenAIEmbeddingModule
from langrila.database.chroma import ChromaCollectionModule, ChromaRetrievalModule

#######################
# create collection
#######################

embedder = OpenAIEmbeddingModule(
        api_key_name="API_KEY", 
    )

collection = ChromaCollectionModule(
    persistence_directory="chroma", 
    collection_name="sample", 
    embedder=embedder
)

documents = [
    "Langrila is a useful tool to use ChatGPT with OpenAI API or Azure in an easy way.",
    "LangChain is a framework for developing applications powered by language models.", 
    "LlamaIndex (GPT Index) is a data framework for your LLM application."
]

collection(documents=documents)

#######################
# retrieval
#######################

# In the case collection was already instantiated
# retriever = collection.as_retriever(n_results=2, threshold_similarity=0.8)

retriever = ChromaRetrievalModule(
            embedder=embedder,
            persistence_directory="chroma", 
            collection_name="sample", 
            n_results=2,
            threshold_similarity=0.8,
        )

query = "What is Langrila?"
retriever(query, where=None).model_dump()

>>> {'ids': ['0'],
 'documents': ['Langrila is a useful tool to use ChatGPT with OpenAI API or Azure in an easy way.'],
 'metadatas': [None],
 'similarities': [0.8371831512206707],
 'usage': {'prompt_tokens': 6, 'completion_tokens': 0}}
```

#### For Qdrant
```python
from langrila import OpenAIEmbeddingModule
from langrila.database.qdrant import QdrantLocalCollectionModule, QdrantLocalRetrievalModule

#######################
# create collection
#######################

embedder = OpenAIEmbeddingModule(
        api_key_name="API_KEY", 
    )

collection = QdrantLocalCollectionModule(
    persistence_directory="qdrant", 
    collection_name="sample", 
    embedder=embedder
)

documents = [
    "Langrila is a useful tool to use ChatGPT with OpenAI API or Azure in an easy way.",
    "LangChain is a framework for developing applications powered by language models.", 
    "LlamaIndex (GPT Index) is a data framework for your LLM application."
]

collection(documents=documents)

#######################
# retrieval
#######################

# In the case collection was already instantiated
# retriever = collection.as_retriever(n_results=2, threshold_similarity=0.8)

retriever = QdrantLocalRetrievalModule(
            embedder=embedder,
            persistence_directory="qdrant", 
            collection_name="sample", 
            n_results=2,
            threshold_similarity=0.8,
        )

query = "What is Langrila?"
retriever(query, filter=None).model_dump()

>>> {'ids': [0],
 'documents': ['Langrila is a useful tool to use ChatGPT with OpenAI API or Azure in an easy way.'],
 'metadatas': [None],
 'similarities': [0.8371831512206701],
 'usage': {'prompt_tokens': 6, 'completion_tokens': 0}}
```

### Specific use case
The library supports a variety of use cases by combining modules such as these and defining new modules. For example, the following is an example of a module that combines basic Retrieval and prompt templates. 

```python
from langrila import BaseModule, PromptTemplate
from langrila.database.chroma import ChromaRetrievalModule


class RetrievalChatWithTemplate(BaseModule):
    def __init__(
        self,
        api_type: str = "azure",
        api_version: str = "2023-07-01-preview",
        api_version_embedding: str = "2023-05-15",
        endpoint_name: str = "ENDPOINT",
        api_key_name: str = "API_KEY",
        deployment_id_name: str = "DEPLOY_ID",
        deployment_id_name_embedding: str = "DEPLOY_ID_EMBEDDING",
        max_tokens: int = 2048,
        timeout: int = 60,
        max_retries: int = 2,
        context_length: int = None,
        model_name="gpt-3.5-turbo-16k-0613",
        path_to_index: str = "path-to-your-chroma-index",
        index_collection_name: str = "your-collection-name"
    ):
        chatmodel_kwargs = {
            "api_type": api_type,
            "api_version": api_version,
            "api_key_name": api_key_name,
            "endpoint_name": endpoint_name,
            "deployment_id_name": deployment_id_name,
        }

        embedding_kwargs = {
            "api_type": api_type,
            "api_version": api_version_embedding,
            "api_key_name": api_key_name,
            "endpoint_name": endpoint_name,
            "deployment_id_name": deployment_id_name_embedding,
            "model_name": "text-embedding-ada-002"
        }

        self.chat = OpenAIChatModule(
            **chatmodel_kwargs,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            model_name=model_name,
            context_length=context_length,
            conversation_memory=None,
        )

        self.retriever = ChromaRetrievalModule(
            embedder=OpenAIEmbeddingModule(**embedding_kwargs),
            persistence_directory=path_to_index,
            collection_name=index_collection_name,
            n_results=5,
            threshold_similarity=0.8,
        )

    def run(self, prompt: str, prompt_template: Optional[PromptTemplate] = None):
        retrieval_results = self.retriever(prompt)
        relevant_docs = "\n\n".join(retrieval_results.documents)

        if isinstance(prompt_template, PromptTemplate):
            prompt = prompt_template.set_args(relevant_docs=relevant_docs).format()

        response = self.chat(prompt)
        return response
```
