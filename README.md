# Langri-La
Langrila is a useful tool to use API-based LLM in an easy way. This library put emphasis on simple architecture for readability.

# Dependencies
## must
- Python >=3.10,<3.13

## as needed
- openai and tiktoken for OpenAI API
- google-generativeai for Gemini API
- qdrant-client, chromadb or usearch for retrieval

# Contribution
## Coding policy
1. Sticking to simplicity : This library is motivated by simplifying architecture for readability. Thus too much abstraction should be avoided.
2. Implementing minimum modules : The more functions each module has, the more complex the source code becomes. Langrila focuses on implementing minimum necessary functions in each module Basically module has only a responsibility expressed by thier module name and main function is implemented in a method easy to understand like `run()` or `arun()` methods except for some.

## Branch management rule
- Topic branch are checkout from main branch.
- Topic branch should be small.

# Installation
## clone
```
git clone git@github.com:taikinman/langrila.git
```

## pip
See [pyproject.toml](./pyproject.toml) for more detailed installation options.

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

# For OpenAI and Usearch
pip install -e .[openai,usearch]

# For All
pip install -e .[all]
```

## poetry
See [pyproject.toml](./pyproject.toml) for more detailed installation options.

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

# For OpenAI and Usearch
poetry add --editable /path/to/langrila/ --extras "openai usearch"

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

# show result
response.model_dump()

>>> {'message': {'role': 'assistant',
  'content': "Establish a consistent bedtime routine and stick to it every night, even on weekends. This will help signal to your body that it's time to wind down and prepare for sleep."},
 'usage': {'model_name': 'gpt-3.5-turbo-0125',
  'prompt_tokens': 21,
  'completion_tokens': 35},
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

# show result
response.model_dump()

>>> {'message': parts {
   text: "**Establish a consistent sleep schedule, going to bed and waking up at the same time every day, even on weekends.** \n"
 }
 role: "model",
 'usage': {'model_name': 'gemini-1.5-flash',
  'prompt_tokens': 15,
  'completion_tokens': 26},
 'prompt': [parts {
    text: "Please give me only one advice to improve the quality of my sleep."
  }
  role: "user"]}
```

### For Gemini on VertexAI
You can also run gemini on VertexAI with a few additional arguments instead of `api_key_env_name`.

```python
from langrila.gemini import GeminiChatModule

chat = GeminiChatModule(
    # api_key_env_name="GEMINI_API_KEY",
    api_type="vertexai",
    project_id_env_name="PROJECT_ID",
    location_env_name="LOCATION",
    model_name="gemini-1.5-flash",
)

prompt = "Please give me only one advice to improve the quality of my sleep."

# synchronous processing
response = chat.run(prompt)

# asynchronous processing
response = await chat.arun(prompt)

# show result
response.model_dump()

>>> {'message': role: "model"
 parts {
   text: "**Establish a consistent sleep schedule, going to bed and waking up at the same time every day, even on weekends.** \n"
 },
 'usage': {'model_name': 'gemini-1.5-flash',
  'prompt_tokens': 14,
  'completion_tokens': 26},
 'prompt': [role: "user"
  parts {
    text: "Please give me only one advice to improve the quality of my sleep."
  }]}
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

response = chat.run(prompt, images=image) # or response = await chat.arun(prompt, images=image)

# show result
response.model_dump()
```

Gemini on VertexAI has the same interface above.

## System instruction
You can pass system instruction to llm like below:

```python
from langrila.openai import OpenAIChatModule

# For conventional model
chat = OpenAIChatModule(
    api_key_env_name="API_KEY",  # env variable name
    model_name="gpt-3.5-turbo-0125",
    system_instruction="You must answer any questions only with yes or no.",
)

prompt = "Please give me only one advice to improve the quality of my sleep."

# synchronous processing
response = chat.run(prompt)

# show result
response.model_dump()

>>> {'message': {'role': 'assistant', 'content': 'Yes.'},
 'usage': {'model_name': 'gpt-3.5-turbo-0125',
  'prompt_tokens': 36,
  'completion_tokens': 2},
 'prompt': [{'role': 'user',
   'content': 'Please give me only one advice to improve the quality of my sleep.'}]}
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
# response = chat.astream(prompt)
# [r async for r in response]

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
# response = model_chat.astream(prompt)
# [r async for r in response]


>>> [CompletionResults(message=parts {
   text: "**"
 }
 role: "model"
 , usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0), prompt=''),
 CompletionResults(message=parts {
   text: "**Establish a consistent sleep schedule, going to bed and waking up at the same time"
 }
 role: "model"
 , usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0), prompt=''),
 CompletionResults(message=parts {
   text: "**Establish a consistent sleep schedule, going to bed and waking up at the same time every day, even on weekends.** \n"
 }
 role: "model"
 , usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0), prompt=''),
 CompletionResults(message=parts {
   text: "**Establish a consistent sleep schedule, going to bed and waking up at the same time every day, even on weekends.** \n"
 }
 role: "model"
 , usage=Usage(prompt_tokens=15, completion_tokens=26, total_tokens=41), prompt=[parts {
   text: "Please give me only one advice to improve the quality of my sleep."
 }
 role: "user"
 ])]
```

## Function Calling

Now we use these functions.

```python
def power_disco_ball(power: bool) -> bool:
    """Powers the spinning disco ball."""
    return f"Disco ball is {'spinning!' if power else 'stopped.'}"


def start_music(energetic: bool, loud: bool, bpm: int) -> str:
    """Play some music matching the specified parameters.

    Args:
      energetic: Whether the music is energetic or not.
      loud: Whether the music is loud or not.
      bpm: The beats per minute of the music.

    Returns: The name of the song being played.
    """
    return f"Starting music! {energetic=} {loud=}, {bpm=}"


def dim_lights(brightness: float) -> bool:
    """Dim the lights.

    Args:
      brightness: The brightness of the lights, 0.0 is off, 1.0 is full.
    """
    return f"Lights are now set to {brightness:.0%}"
```

### For OpenAI
```python
from langrila.openai import OpenAIFunctionCallingModule, ToolConfig, ToolParameter, ToolProperty

tool_configs = [
    ToolConfig(
        name="power_disco_ball",
        description="Powers the spinning disco ball.",
        parameters=ToolParameter(
            properties=[
                ToolProperty(
                    name="power",
                    type="boolean",
                    description="Boolean to spin disco ball.",
                ),
            ],
            required=["power"],
        ),
    ),
    ToolConfig(
        name="start_music",
        description="Play some music matching the specified parameters.",
        parameters=ToolParameter(
            properties=[
                ToolProperty(
                    name="energetic",
                    type="boolean",
                    description="Whether the music is energetic or not.",
                ),
                ToolProperty(
                    name="loud", type="boolean", description="Whether the music is loud or not."
                ),
                ToolProperty(
                    name="bpm", type="number", description="The beats per minute of the music."
                ),
            ],
            required=["energetic", "loud", "bpm"],
        ),
    ),
    ToolConfig(
        name="dim_lights",
        description="Dim the lights.",
        parameters=ToolParameter(
            properties=[
                ToolProperty(
                    name="brightness",
                    type="number",
                    description="The brightness of the lights, 0.0 is off, 1.0 is full.",
                ),
            ],
            required=["brightness"],
        ),
    ),
]

function_calling = OpenAIFunctionCallingModule(
    api_key_env_name="API_KEY",
    model_name="gpt-4o-2024-05-13",
    tools=[power_disco_ball, start_music, dim_lights],
    tool_configs=tool_configs,
)

prompt = "Turn this place into a party!"
response = await function_calling.arun(
    prompt,
    # If you want to use a specific tool, you can specify it here
    # tool_choice="power_disco_ball",
)

# Show result
response.model_dump()

>>> {'usage': {'model_name': 'gpt-4o-2024-05-13',
  'prompt_tokens': 158,
  'completion_tokens': 75},
 'results': [{'call_id': 'call_JihPSqJEgAXZqExPMaWTGT60',
   'funcname': 'power_disco_ball',
   'args': '{"power": true}',
   'output': 'Disco ball is spinning!'},
  {'call_id': 'call_BS2IksJBeLkTgt7qghgH5lPt',
   'funcname': 'start_music',
   'args': '{"energetic": true, "loud": true, "bpm": 120}',
   'output': 'Starting music! energetic=True loud=True, bpm=120'},
  {'call_id': 'call_XELGql9jojBILNkH4JVtZUu9',
   'funcname': 'dim_lights',
   'args': '{"brightness": 0.3}',
   'output': 'Lights are now set to 30%'}],
 'prompt': [{'role': 'user',
   'content': 'Turn this place into a party!',
   'name': None}]}
```

### For Azure
Basically same interface with For OpenAI.

```python
function_calling = OpenAIFunctionCallingModule(
            api_key_env_name = "AZURE_API_KEY", # env variable name
            model_name="gpt-4o-2024-05-13",
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
from langrila.gemini import GeminiFunctionCallingModule
from langrila.gemini.genai.tools import ToolConfig, ToolParameter, ToolProperty

# tool_configs is the same as that of OpenAI
tool_configs = [
    ToolConfig(
        name="power_disco_ball",
        description="Powers the spinning disco ball.",
        parameters=ToolParameter(
            properties=[
                ToolProperty(
                    name="power",
                    type="boolean",
                    description="Boolean to spin disco ball.",
                ),
            ],
            required=["power"],
        ),
    ),
    ToolConfig(
        name="start_music",
        description="Play some music matching the specified parameters.",
        parameters=ToolParameter(
            properties=[
                ToolProperty(
                    name="energetic",
                    type="boolean",
                    description="Whether the music is energetic or not.",
                ),
                ToolProperty(
                    name="loud", type="boolean", description="Whether the music is loud or not."
                ),
                ToolProperty(
                    name="bpm", type="number", description="The beats per minute of the music."
                ),
            ],
            required=["energetic", "loud", "bpm"],
        ),
    ),
    ToolConfig(
        name="dim_lights",
        description="Dim the lights.",
        parameters=ToolParameter(
            properties=[
                ToolProperty(
                    name="brightness",
                    type="number",
                    description="The brightness of the lights, 0.0 is off, 1.0 is full.",
                ),
            ],
            required=["brightness"],
        ),
    ),
]

function_calling = GeminiFunctionCallingModule(
    api_key_env_name="GEMINI_API_KEY",
    model_name="gemini-1.5-flash",
    tools=[power_disco_ball, start_music, dim_lights],
    tool_configs=tool_configs,
)

prompt = "Turn this place into a party!"
response = await function_calling.arun(
    prompt,
    # If you want to use a specific tool, you can specify it here
    # tool_choice=["power_disco_ball"],
)

# show result
response.model_dump()

>>> {'usage': {'model_name': 'gemini-1.5-flash',
  'prompt_tokens': 8,
  'completion_tokens': 58},
 'results': [{'call_id': None,
   'funcname': 'power_disco_ball',
   'args': '{"power": true}',
   'output': 'Disco ball is spinning!'},
  {'call_id': None,
   'funcname': 'start_music',
   'args': '{"loud": true, "energetic": true, "bpm": 120.0}',
   'output': 'Starting music! energetic=True loud=True, bpm=120.0'},
  {'call_id': None,
   'funcname': 'dim_lights',
   'args': '{"brightness": 0.5}',
   'output': 'Lights are now set to 50%'}],
 'prompt': [parts {
    text: "Turn this place into a party!"
  }
  role: "user"]}
```

For VertexAI, things you only do is to replace genai modules with vertexai modules and then pass a few additional arguments instead of `api_key_env_name` as well as `GeminiChatModule`. Here is an instruction.

```python
# from langrila.gemini.genai.tools import ToolConfig, ToolParameter, ToolProperty
from langrila.gemini.vertexai.tools import ToolConfig, ToolParameter, ToolProperty

function_calling = GeminiFunctionCallingModule(
    # api_key_env_name="GEMINI_API_KEY",
    api_type="vertexai",
    project_id_env_name="PROJECT_ID",
    location_env_name="LOCATION",
    model_name="gemini-1.5-flash",
    tools=[power_disco_ball, start_music, dim_lights],
    tool_configs=tool_configs,
)

```

## Total token counting
Total number of tokens can be summed for each models by `TokenCounter`. It's useful to see cost when multiple models are cooperatively working.

```python
from langrila import TokenCounter
from langrila.openai import OpenAIChatModule

# initialize shared token counter
token_counter = TokenCounter()

# For conventional model
chat1 = OpenAIChatModule(
    api_key_env_name="API_KEY",  # env variable name
    model_name="gpt-3.5-turbo-0125",
    token_counter=token_counter,
    # organization_id_env_name="ORGANIZATION_ID", # env variable name
)

chat2 = OpenAIChatModule(
    api_key_env_name="API_KEY",  # env variable name
    model_name="gpt-4o-2024-05-13",
    token_counter=token_counter,
    # organization_id_env_name="ORGANIZATION_ID", # env variable name
)


prompt = "Please give me only one advice to improve the quality of my sleep."

# generate response
response1 = chat1.run(prompt)
response1 = chat1.run(prompt) # second call to see summed token result
response2 = chat2.run(prompt)

print(token_counter)

>>> {'gpt-3.5-turbo-0125': Usage(prompt_tokens=42, completion_tokens=61, total_tokens=103), 'gpt-4o-2024-05-13': Usage(prompt_tokens=21, completion_tokens=40, total_tokens=61)}
```

## Conversation memory
You can use 2 conversation memory modules by default named InMemoryConversationMemory and JSONConversationMemory. Additionally, langrila supports Cosmos DB and Amazon S3 to store conversation history.

### Conversation memory with local json file
In this example, we specify JSONConversationMemory. 

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

# show result
response.model_dump()

>>> {'message': {'role': 'assistant',
  'content': 'Yes, Rude is a character in Final Fantasy 7. He is a member of the Turks, a group of elite operatives working for the Shinra Electric Power Company. Rude is known for his calm and collected demeanor, as well as his impressive physical strength. He often works alongside his partner, Reno, and is skilled in hand-to-hand combat.'},
 'usage': {'model_name': 'gpt-3.5-turbo-0125',
  'prompt_tokens': 22,
  'completion_tokens': 72},
 'prompt': [{'role': 'user',
   'content': 'Do you know Rude who is the character in Final Fantasy 7.'}]}

# second prompt
message = "What does he think about Tifa?"
response = chat.run(message)

# show result
response.model_dump()

>>> {'message': {'role': 'assistant',
  'content': "In Final Fantasy 7, Rude is shown to have a crush on Tifa Lockhart, one of the main protagonists of the game. He admires her strength and beauty, and is often seen blushing or acting shy around her. Despite being a member of the Turks and having conflicting interests with Tifa and her friends, Rude's feelings for her show a more human and compassionate side to his character."},
 'usage': {'model_name': 'gpt-3.5-turbo-0125',
  'prompt_tokens': 110,
  'completion_tokens': 84},
 'prompt': [{'role': 'user',
   'content': 'Do you know Rude who is the character in Final Fantasy 7.'},
  {'role': 'assistant',
   'content': 'Yes, Rude is a character in Final Fantasy 7. He is a member of the Turks, a group of elite operatives working for the Shinra Electric Power Company. Rude is known for his calm and collected demeanor, as well as his impressive physical strength. He often works alongside his partner, Reno, and is skilled in hand-to-hand combat.'},
  {'role': 'user', 'content': 'What does he think about Tifa?'}]}
```

`OpenAIFunctionCallingModule`, `GeminiChatModule` and `GeminiFunctionCallingModule` also allow you to save conversation in JSON format.

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

# show result
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
 'usage': {'model_name': 'text-embedding-3-large',
  'prompt_tokens': 14,
  'completion_tokens': 0}}
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
Now langrila supports qdrant, chroma and usearch for retrieval.

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

query = "What is Langrila?"
retrieval_reuslt = retriever.run(query, filter=None)

# show result
retrieval_result.model_dump()

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

query = "What is Langrila?"
retrieval_result = retriever.run(query, filter=None)

# show result
retrieval_result.model_dump()

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

### For Usearch
Usearch originally doesn't support metadata storing and filtering, so in langrila, those functions are realized by SQLite3 and postprocessing.

```python
from langrila.database.usearch import UsearchLocalCollectionModule, UsearchLocalRetrievalModule
from langrila.openai import OpenAIEmbeddingModule

#######################
# create collection
#######################

embedder = OpenAIEmbeddingModule(
    api_key_env_name="API_KEY",
    model_name="text-embedding-3-small",
    dimensions=1536,
)

collection = UsearchLocalCollectionModule(
    persistence_directory="./usearch_test",
    collection_name="sample",
    embedder=embedder,
    dtype = "f16",
    ndim = 1536,
    connectivity = 16,
    expansion_add = 128,
    expansion_search = 64,
)

documents = [
    "Langrila is a useful tool to use ChatGPT with OpenAI API or Azure in an easy way.",
    "LangChain is a framework for developing applications powered by language models.",
    "LlamaIndex (GPT Index) is a data framework for your LLM application.",
]

# Strongly recommended because search result may be different when new vectors are inserted after existing vectors are removed.
# Instead, rebuilding the index is recommended using `delete_collection` before upserting.
# Or use exact search to avoid this issue when search time.
collection.delete_collection()

collection.run(documents=documents) # metadatas could also be used. 

# #######################
# # retrieval
# #######################

# In the case collection was already instantiated
# retriever = collection.as_retriever(n_results=2, threshold_similarity=0.5)

retriever = UsearchLocalRetrievalModule(
    embedder=embedder,
    persistence_directory="./usearch_test",
    collection_name="sample",
    dtype = "f16",
    ndim=1536,
    connectivity = 16,
    expansion_add = 128,
    expansion_search = 64,
    n_results=2,
    score_threshold=0.5,
)

query = "What is Langrila?"
retrieval_result = retriever.run(query, filter=None, exact=False)

# show result
retrieval_result.model_dump()

>>> {'ids': [0],
 'documents': ['Langrila is a useful tool to use ChatGPT with OpenAI API or Azure in an easy way.'],
 'metadatas': [{'document': 'Langrila is a useful tool to use ChatGPT with OpenAI API or Azure in an easy way.'}],
 'scores': [0.46986961364746094],
 'collections': ['sample'],
 'usage': {'prompt_tokens': 6, 'completion_tokens': 0}}
```

When you need to filter retrieval results by metadata in search time, you can implement your custom metadata filter. Base class of metadata filter is in [base.py](src/langrila/base.py). For more details, see : [usearch.py](src/langrila/database/usearch.py).

### Specific use case
The library supports a variety of use cases by combining modules such as these and defining new modules. For example, the following is an example of a module that combines basic Retrieval and prompt templates. 
