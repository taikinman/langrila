# Langri-La
Langrila is a useful tool to use API-based LLM in an easy way. This library put emphasis on simple architecture for readability.

# Dependencies
## must
- Python >=3.10,<3.13

## as needed
- openai and tiktoken for OpenAI API
- google-generativeai for Gemini API
- qdrant-client or chromadb for retrieval

# Contribution
## Coding policy
1. Sticking to simplicity : This library is motivated by simplifying architecture for readability. Thus multiple inheriting and nested inheriting should be avoided as much as possible for basic modules at least.
2. Making responsibility Independent : The responsibility of each module must be closed into each module itself. It means a module is not allowed to affect other modules.
3. Implementing minimum modules : The more functions each module has, the more complex the source code becomes. Langrila focuses on implementing minimum necessary functions in each module.

## Branch management rule
- Topic branch are checkout from main branch.
- Topic branch should be small.

# Installation
## clone
```
git clone git@github.com:taikinman/langrila.git
```

## pip
```
cd langrila

# For OpenAI
pip install -e .[openai]

# For Gemini
pip install -e .[gemini]

# For both
pip install -e .[openai,gemini]

# For OpenAI and Qdrant
pip install -e .[openai,qdrant]

# For OpenAI and Chroma
pip install -e .[openai,chroma]

# For All
pip install -e .[all]
```

## poetry
```
# For OpenAI
poetry add --editable /path/to/langrila/ --extras openai

# For Gemini
poetry add --editable /path/to/langrila/ --extras gemini

# For both
poetry add --editable /path/to/langrila/ --extras "openai gemini"

# For OpenAI and Qdrant
poetry add --editable /path/to/langrila/ --extras "openai qdrant"

# For OpenAI and Chroma
poetry add --editable /path/to/langrila/ --extras "openai chroma"

# For all extra dependencies
poetry add --editable /path/to/langrila/ --extras all
```

# Pre-requirement
1. Pre-configure environment variables to use OpenAI API, Azure OpenAI Service or Gemini API.

# Supported models for OpenAI
## Chat models
- gpt-3.5-turbo-0301
- gpt-3.5-turbo-0613
- gpt-3.5-turbo-16k-0613
- gpt-3.5-turbo-instruct
- gpt-3.5-turbo-1106
- gpt-3.5-turbo-0125
- gpt-4-0314
- gpt-4-0613
- gpt-4-32k-0314
- gpt-4-32k-0613
- gpt-4-1106-preview
- gpt-4-vision-preview
- gpt-4-0125-preview
- gpt-4-turbo-2024-04-09
- gpt-4o-2024-05-13

## Embedding models
- text-embedding-ada-002
- text-embedding-3-small
- text-embedding-3-large

## Aliases
```
{'gpt-4o': 'gpt-4o-2024-05-13',
 'gpt-4-turbo': 'gpt-4-turbo-2024-04-09',
 'gpt-3.5-turbo': 'gpt-3.5-turbo-0125'}
```

# Supported models for Gemini
## Chat models
- gemini-1.5-pro
- gemini-1.5-flash

# Breaking changes
<details>
<summary>v0.0.7 -> v0.0.8</summary>

Database modules has breaking changes from v0.0.7 to v0.0.8 such as rename of method, change the interface of some methods. For more details, see [this PR](https://github.com/taikinman/langrila/pull/34).

</details>


<details>
<summary>v0.0.2 -> v0.0.3</summary>

I have integrated gemini api into langrila on v0.0.3. When doing this, the modules for openai and azure openai should be separated from gemini's modules so that unnecessary dependencies won't happen while those components has the same interface. But migration is easy. Basically only thing to do is to change import modules level like `from langrila import OpenAIChatModule` to `from langrila.openai import OpenAIChatModule`. It's the same as other modules related to openai api.

Second change point is return type of `stream()` and `astream()` method. From v0.0.3, all return types of all chunks is CompletionResults.

Third point is the name of results class : `RetrievalResult` to `RetrievalResults`. `RetrievalResults` model has collections atribute now. Also similarities was replaced with scores.

</details>

# Basic usage
## Basic example
### For OpenAI
```python
from langrila.openai import OpenAIChatModule

# For conventional model
chat = OpenAIChatModule(
    api_key_env_name = "API_KEY", # env variable name
    model_name="gpt-3.5-turbo-0125",
    # organization_id_env_name="ORGANIZATION_ID", # env variable name
)

prompt = "Please give me only one advice to improve the quality of my sleep."

# synchronous processing
response = chat.run(prompt)

# asynchronous processing
response = await chat.arun(prompt)

response.model_dump()

>>> {'message': {'role': 'assistant',
  'content': 'Establish a consistent bedtime routine and stick to it every night, including going to bed and waking up at the same time each day.'},
 'usage': {'prompt_tokens': 21, 'completion_tokens': 26},
 'prompt': [{'role': 'user',
   'content': 'Please give me only one advice to improve the quality of my sleep.'}]}
```

### For Azure
If you specify the arguments likes below, you can get the response from Azure OpenAI.
```python
chat = OpenAIChatModule(
        api_key_env_name="AZURE_API_KEY", # env variable name
        model_name="gpt-3.5-turbo-0125",
        api_type="azure",
        api_version="2024-05-01-preview", 
        deployment_id_env_name="DEPLOY_ID", # env variable name
        endpoint_env_name="ENDPOINT", # env variable name
    )
```

### For Gemini
You can use gemini in the same interface.
```python
from langrila.gemini import GeminiChatModule

chat = GeminiChatModule(
    api_key_env_name="GEMINI_API_KEY",
    model_name="gemini-1.5-flash",
)

prompt = "Please give me only one advice to improve the quality of my sleep."

# synchronous processing
response = chat.run(prompt)

# asynchronous processing
response = await chat.arun(prompt)

response.model_dump()

>>> {'message': {'role': 'model',
  'parts': ['**Establish a consistent sleep schedule, going to bed and waking up at the same time every day, even on weekends.** \n']},
 'usage': {'prompt_tokens': 15, 'completion_tokens': 26},
 'prompt': [{'role': 'user',
   'parts': ['Please give me only one advice to improve the quality of my sleep.']}]}
```

## Vision model
### For OpenAI
You can pass image data to chat module. (For Azure OpenAI, only different from the arguments when instantiation of chat module)
```python
from PIL import Image

# In this example, I use a picture of "osechi-ryori"
image = Image.open("path/to/your/local/image/file")

chat = OpenAIChatModule(
    api_key_env_name="API_KEY",
    model_name="gpt-4o-2024-05-13",
)

# stream, astream are also runnable
prompt = "What kind of food is in the picture?"
response = await chat.arun(prompt, images=image) # multiple image input is also allowed
```

### For Gemini
```python
chat = GeminiChatModule(
    api_key_env_name="GEMINI_API_KEY",
    model_name="gemini-1.5-flash",
)

response = await chat.arun(prompt, images=image) # or response = await chat.arun(prompt, images=image)
response.model_dump()
```


## Batch processing
For optimization, you can process multiple prompts as batches.
```python
prompts = [
    "Please give me only one advice to improve the quality of my sleep.", 
    "Please give me only one advice to improve my memory.",
    "Please give me only one advice on how to make exercise a habit.",
    "Please give me only one advice to help me not get bored with things so quickly." 
            ]

await chat.abatch_run(prompts, batch_size=4)
```

## Stream
### For OpenAI
```python
from langrila.openai import OpenAIChatModule

chat = OpenAIChatModule(
    api_key_env_name = "API_KEY", # env variable name
    model_name="gpt-3.5-turbo-0125",
    # organization_id_env_name="ORGANIZATION_ID", # env variable name
)

prompt = "Please give me only one advice to improve the quality of my sleep."
response = chat.stream(prompt)
list(response)


# For async process
response = chat.astream(prompt)
[r async for r in response]

>>> [CompletionResults(message={'role': 'assistant', 'content': ''}, usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0), prompt=[{}]),
 CompletionResults(message={'role': 'assistant', 'content': 'Establish'}, usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0), prompt=[{}]),
 CompletionResults(message={'role': 'assistant', 'content': 'Establish a'}, usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0), prompt=[{}]),
 CompletionResults(message={'role': 'assistant', 'content': 'Establish a consistent'}, usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0), prompt=[{}]),
 CompletionResults(message={'role': 'assistant', 'content': 'Establish a consistent bedtime'}, usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0), prompt=[{}]),
 CompletionResults(message={'role': 'assistant', 'content': 'Establish a consistent bedtime routine'}, usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0), prompt=[{}]),
 CompletionResults(message={'role': 'assistant', 'content': 'Establish a consistent bedtime routine and'}, usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0), prompt=[{}]),
...
 CompletionResults(message={'role': 'assistant', 'content': "Establish a consistent bedtime routine and stick to it every night, even on weekends. This can help signal to your body that it's time to wind down and prepare for sleep."}, usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0), prompt=[{}]),
 CompletionResults(message={'role': 'assistant', 'content': "Establish a consistent bedtime routine and stick to it every night, even on weekends. This can help signal to your body that it's time to wind down and prepare for sleep."}, usage=Usage(prompt_tokens=21, completion_tokens=42, total_tokens=63), prompt=[{'role': 'user', 'content': 'Please give me only one advice to improve the quality of my sleep.'}])] # at the end of stream, model returns entire response and usage
```

For Azure OpenAI, inteface is the same.

### For Gemini
```python
from langrila.gemini import GeminiChatModule

model_chat = GeminiChatModule(
    api_key_env_name="GEMINI_API_KEY",
    model_name="gemini-1.5-flash",
)

prompt = "Please give me only one advice to improve the quality of my sleep."

response = model_chat.stream(prompt)
list(response)

# For async process
response = model_chat.astream(prompt)
[r async for r in response]


>>> [CompletionResults(message={'role': 'model', 'parts': ['**']}, usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0), prompt=''),
 CompletionResults(message={'role': 'model', 'parts': ['**Establish a consistent sleep schedule, going to bed and waking up around the same time']}, usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0), prompt=''),
 CompletionResults(message={'role': 'model', 'parts': ['**Establish a consistent sleep schedule, going to bed and waking up around the same time each day, even on weekends.** \n']}, usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0), prompt=''),
 CompletionResults(message={'role': 'model', 'parts': ['**Establish a consistent sleep schedule, going to bed and waking up around the same time each day, even on weekends.** \n']}, usage=Usage(prompt_tokens=15, completion_tokens=26, total_tokens=41), prompt=[{'role': 'user', 'parts': ['Please give me only one advice to improve the quality of my sleep.']}, {'role': 'model', 'parts': ['**Establish a consistent sleep schedule, going to bed and waking up around the same time each day, even on weekends.** \n']}])]
```

## Function Calling
### For OpenAI
```python
from langrila.openai import OpenAIFunctionCallingModule, ToolConfig, ToolParameter, ToolProperty


def get_weather(city: str, date: str) -> str:
    return f"The weather in {city} on {date} is sunny."


tool_config = ToolConfig(
    name="get_weather",
    description="Get weather at current location.",
    parameters=ToolParameter(
        properties=[
            ToolProperty(name="city", type="string", description="City name"),
            ToolProperty(name="date", type="string", description="Date"),
        ],
        required=["city", "date"],
    ),
)


function_calling = OpenAIFunctionCallingModule(
    api_key_env_name="API_KEY",
    model_name="gpt-3.5-turbo-0125",
    tools=[get_weather],
    tool_configs=[tool_config],
)

response = await function_calling.arun("What is the weather in New York on 2022-01-01?")
response.model_dump()

>>> {'usage': {'prompt_tokens': 72, 'completion_tokens': 24},
 'results': [{'call_id': 'call_Yg5mU4wLX6nFkgVnlEaMp1mC',
   'funcname': 'get_weather',
   'args': '{"city":"New York","date":"2022-01-01"}',
   'output': 'The weather in New York on 2022-01-01 is sunny.'}],
 'prompt': [{'role': 'user',
   'content': 'What is the weather in New York on 2022-01-01?'}]}

```

### For Azure
Basically same interface with For OpenAI.
```python
function_calling = OpenAIFunctionCallingModule(
            api_key_env_name = "AZURE_API_KEY", # env variable name
            model_name="gpt-3.5-turbo-0125",
            api_type="azure", 
            api_version="2024-05-01-preview",
            endpoint_env_name="ENDPOINT", # env variable name
            deployment_id_env_name="DEPLOY_ID", # env variable name
            timeout=60,
            max_retries=2,
            tools=[get_weather], 
            tool_configs=[tool_config],
        )
```

### For Gemini
For Gemini, only things you have to do is to replace module.
```python
from langrila.gemini import GeminiFunctionCallingModule, ToolConfig, ToolParameter, ToolProperty


def get_weather(city: str, date: str) -> str:
    return f"The weather in {city} on {date} is sunny."


tool_config = ToolConfig(
    name="get_weather",
    description="Get weather at current location.",
    parameters=ToolParameter(
        properties=[
            ToolProperty(name="city", type="string", description="City name"),
            ToolProperty(name="date", type="string", description="Date"),
        ],
        required=["city", "date"],
    ),
)

function_calling = GeminiFunctionCallingModule(
    api_key_env_name="GEMINI_API_KEY",
    model_name="gemini-1.5-flash",
    tools=[get_weather],
    tool_configs=[tool_config],
)

prompt = "What is the weather in New York on 2022-01-01?"
response = await function_calling.arun(prompt)
response.model_dump()

>>> {'usage': {'prompt_tokens': 21, 'completion_tokens': 30},
 'results': [{'call_id': None,
   'funcname': 'get_weather',
   'args': '{"date": "2022-01-01", "city": "New York"}',
   'output': 'The weather in New York on 2022-01-01 is sunny.'}],
 'prompt': [{'role': 'user',
   'parts': ['What is the weather in New York on 2022-01-01?']}]}
```

## Conversation memory
### For standard chat model
```python
from langrila import JSONConversationMemory

chat = OpenAIChatModule(
    api_key_env_name="API_KEY",  # env variable name
    model_name="gpt-3.5-turbo-0125",
    conversation_memory=JSONConversationMemory("./conversation_memory.json"),
    timeout=60,
    max_retries=2,
)


message = "Do you know Rude who is the character in Final Fantasy 7."
response = chat.run(message)
response.model_dump()
>>> {'message': {'role': 'assistant',
  'content': 'Yes, Rude is a character in Final Fantasy 7. He is a member of the Turks, a group of elite operatives working for the Shinra Electric Power Company. Rude is known for his calm and collected demeanor, as well as his impressive physical strength. He often works alongside his partner, Reno, and is skilled in hand-to-hand combat.'},
 'usage': {'prompt_tokens': 22, 'completion_tokens': 72},
 'prompt': [{'role': 'user',
   'content': 'Do you know Rude who is the character in Final Fantasy 7.'}]}

# second prompt
message = "What does he think about Tifa?"
response = chat.run(message)
response.model_dump()
>>> {'message': {'role': 'assistant',
  'content': "In Final Fantasy 7, Rude is shown to have a respectful and professional relationship with Tifa Lockhart, one of the main protagonists of the game. While Rude is a member of the Turks and initially opposes Tifa and her allies, he does not harbor any personal animosity towards her. In fact, there are moments in the game where Rude shows a sense of admiration for Tifa's strength and determination. Overall, Rude's interactions with Tifa are characterized by mutual respect and a sense of professionalism."},
 'usage': {'prompt_tokens': 110, 'completion_tokens': 106},
 'prompt': [{'role': 'user',
   'content': 'Do you know Rude who is the character in Final Fantasy 7.'},
  {'role': 'assistant',
   'content': 'Yes, Rude is a character in Final Fantasy 7. He is a member of the Turks, a group of elite operatives working for the Shinra Electric Power Company. Rude is known for his calm and collected demeanor, as well as his impressive physical strength. He often works alongside his partner, Reno, and is skilled in hand-to-hand combat.'},
  {'role': 'user', 'content': 'What does he think about Tifa?'}]}
```

### Conversation memory with Cosmos DB
```python
from langrila.openai import OpenAIChatModule
from langrila.memory.cosmos import CosmosConversationMemory

chat = OpenAIChatModule(
    ...
    conversation_memory=CosmosConversationMemory(
        endpoint_env_name = "COSMOS_ENDPOINT", # env variable names
        key_env_name = "COSMOS_KEY", 
        db_env_name = "COSMOS_DB_NAME", 
        container_env_name = "COSMOS_CONTAINER_NAME"
        ),
)
```

### Conversation memory with Amazon S3
```python
from langrila.openai import OpenAIChatModule
from langrila.memory.s3 import S3ConversationMemory

# S3ConversationMemory utilizes `boto3.client("s3")` for its operations.
# configuration and credentials are handled in the same manner.
chat = OpenAIChatModule(
    ...
    conversation_memory=S3ConversationMemory(
        bucket="S3_BUCKET_NAME",
        object_key="OBJECT_KEY", # "PREFIX/OBJECT_KEY" for using prefix
    )
)
```

## Embedding
```python
from langrila.openai import OpenAIEmbeddingModule

embedder = OpenAIEmbeddingModule(
    api_key_env_name="API_KEY",  # env variable name
    model_name="text-embedding-3-large",
)

message = "Please give me only one advice to improve the quality of my sleep."

results = embedder.run(message)
results.model_dump()

>>> {'text': ['Please give me only one advice to improve the quality of my sleep.'],
 'embeddings': [[0.03542472422122955,
   -0.019869724288582802,
   -0.015472551807761192,
   -0.021917158737778664,
   0.017203938215970993,
   0.010305874049663544,
   0.019127702340483665,
   ...]],
 'usage': {'prompt_tokens': 14, 'completion_tokens': 0}}
```

## PromptTemplate
You can manage your prompt as a prompt template that is often used.
```python
from langrila import PromptTemplate

template = PromptTemplate(
    template="""# INSTRUCTIONS
Please answer the following question written in the QUESTION section.

# QUESTION
{question}

# ANSWER
"""
)

question = "Do you know which is more popular in Japan, Takenoko-no-sato and Kinoko-no-yama?"

template.set_args(question=question)

print(template.format())

>>> # INSTRUCTIONS
Please answer the following question written in the QUESTION section.

# QUESTION
Do you know which is more popular in Japan, Takenoko-no-sato and Kinoko-no-yama?

# ANSWER
```

Also prompt template can load from text file.

```python
template = PromptTemplate.from_text_file("./prompt_template.txt")

question = "Do you know which is more popular in Japan, Takenoko-no-sato and Kinoko-no-yama?"

template.set_args(question=question)

print(template.format())

>>> # same output
```

## Retrieval
Now only Qdrant are supported for basic retrieval.

### For Qdrant
```python
from qdrant_client import models

from langrila.database.qdrant import QdrantLocalCollectionModule, QdrantLocalRetrievalModule
from langrila.openai import OpenAIEmbeddingModule

#######################
# create collection
#######################

embedder = OpenAIEmbeddingModule(
    api_key_env_name="API_KEY",
    model_name="text-embedding-3-small",
    dimensions=1536,
)

collection = QdrantLocalCollectionModule(
    persistence_directory="./qdrant_test",
    collection_name="sample",
    embedder=embedder,
    vectors_config=models.VectorParams(
        size=1536,
        distance=models.Distance.COSINE,
    ),
)

documents = [
    "Langrila is a useful tool to use ChatGPT with OpenAI API or Azure in an easy way.",
    "LangChain is a framework for developing applications powered by language models.",
    "LlamaIndex (GPT Index) is a data framework for your LLM application.",
]

collection.run(documents=documents) # metadatas could also be used

# #######################
# # retrieval
# #######################

# In the case collection was already instantiated
# retriever = collection.as_retriever(n_results=2, threshold_similarity=0.5)

retriever = QdrantLocalRetrievalModule(
    embedder=embedder,
    persistence_directory="./qdrant_test",
    collection_name="sample",
    n_results=2,
    score_threshold=0.5,
)

# If multiple collections are available, search all collections and merge each results later, then return top-k results.
query = "What is Langrila?"
retriever.run(query, filter=None).model_dump()

>>> {'ids': [0],
 'documents': ['Langrila is a useful tool to use ChatGPT with OpenAI API or Azure in an easy way.'],
 'metadatas': [{'document': 'Langrila is a useful tool to use ChatGPT with OpenAI API or Azure in an easy way.'}],
 'scores': [0.5303465176248179],
 'collections': ['sample'],
 'usage': {'prompt_tokens': 6, 'completion_tokens': 0}}
```

Qdrant server is also supported by `QdrantRemoteCollectionModule` and `QdrantRemoteRetrievalModule`. Here is a basic example using docker which app container and qdrant container are bridged by same network.

```python
from qdrant_client import models

from langrila.database.qdrant import QdrantRemoteCollectionModule, QdrantRemoteRetrievalModule
from langrila.openai import OpenAIEmbeddingModule

#######################
# create collection
#######################

embedder = OpenAIEmbeddingModule(
    api_key_env_name="API_KEY",
    model_name="text-embedding-3-small",
    dimensions=1536,
)

collection = QdrantRemoteCollectionModule(
    url="http://qdrant",
    port="6333",
    collection_name="sample",
    embedder=embedder,
    vectors_config=models.VectorParams(
        size=1536,
        distance=models.Distance.COSINE,
    ),
)

```

For more details, see [qdrant.py](src/langrila/database/qdrant.py).

### For Chroma
```python
from langrila.database.chroma import ChromaLocalCollectionModule, ChromaLocalRetrievalModule
from langrila.openai import OpenAIEmbeddingModule

#######################
# create collection
#######################

embedder = OpenAIEmbeddingModule(
    api_key_env_name="API_KEY",
    model_name="text-embedding-3-small",
    dimensions=1536,
)

collection = ChromaLocalCollectionModule(
    persistence_directory="./chroma_test",
    collection_name="sample",
    embedder=embedder,
)

documents = [
    "Langrila is a useful tool to use ChatGPT with OpenAI API or Azure in an easy way.",
    "LangChain is a framework for developing applications powered by language models.",
    "LlamaIndex (GPT Index) is a data framework for your LLM application.",
]

collection.run(documents=documents) # metadatas could also be used

# #######################
# # retrieval
# #######################

# In the case collection was already instantiated
# retriever = collection.as_retriever(n_results=2, threshold_similarity=0.5)

retriever = ChromaLocalRetrievalModule(
    embedder=embedder,
    persistence_directory="./chroma_test",
    collection_name="sample",
    n_results=2,
    score_threshold=0.5,
)

# If multiple collections are available, search all collections and merge each results later, then return top-k results.
query = "What is Langrila?"
retriever.run(query, filter=None).model_dump()

>>> {'ids': [0],
 'documents': ['Langrila is a useful tool to use ChatGPT with OpenAI API or Azure in an easy way.'],
 'metadatas': [{'document': 'Langrila is a useful tool to use ChatGPT with OpenAI API or Azure in an easy way.'}],
 'scores': [0.46960276455443584],
 'collections': ['sample'],
 'usage': {'prompt_tokens': 6, 'completion_tokens': 0}}
```

HttpClient is also supported by `ChromaRemoteCollectionModule` and `ChromaRemoteRetrievalModule`. Here is a basic example using docker which app container and chroma container are bridged by same network.

```python
from langrila.database.chroma import ChromaRemoteCollectionModule
from langrila.openai import OpenAIEmbeddingModule

#######################
# create collection
#######################

embedder = OpenAIEmbeddingModule(
    api_key_env_name="API_KEY",
    model_name="text-embedding-3-small",
    dimensions=1536,
)

collection = ChromaRemoteCollectionModule(
    host="chroma",
    port="8000",
    collection_name="sample",
    embedder=embedder,
)
```

For more details, see [chroma.py](src/langrila/database/chroma.py).

### Specific use case
The library supports a variety of use cases by combining modules such as these and defining new modules. For example, the following is an example of a module that combines basic Retrieval and prompt templates. 
