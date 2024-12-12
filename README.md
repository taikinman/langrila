# Langri-La
Langrila is an open-source third-party python package that is useful to use API-based LLM in the same interface. This package puts emphasis on simple architecture for readability. This package is just personal project.

# Contribution
## Coding style
1. Sticking to simplicity : This library is motivated by simplifying architecture for readability. Thus too much abstraction should be avoided.
2. Implementing minimum modules : The more functions each module has, the more complex the source code becomes. Langrila focuses on implementing minimum necessary functions in each module Basically module has only a responsibility expressed by thier module name and main function is implemented in a method easy to understand like `run()` or `arun()` methods except for some.

## Branch management rule
- Topic branch are checkout from main branch.
- Topic branch should be small.

# Prerequisites
If necessary, set environment variables to use OpenAI API, Azure OpenAI Service, Gemini API, and Claude API; if using VertexAI or Amazon Bedrock, check each platform's user guide and authenticate in advance VertexAI and Amazon Bedrock.

# Supported llm client
- OpenAI
- Azure OpenAI
- Gemini on Google AI Studio
- Gemini on VertexAI
- Claude on Anthropic
- Claude on Amazon Bedrock
- Claude on VertexAI (not tested)

# Breaking changes
<details>
<summary>v0.0.20 -> v0.1.0</summary>

Interface was updated in v0.1.0, and response format is different from before. Main difference is how to access response text in completion results. 

Before: 
```python
response.message["content"]
```

In v0.1.0:
```python
response.message.content[0].text # for single completion
```


For more details, please see [introduction notebook](https://github.com/taikinman/langrila/blob/main/notebooks/01.introduction.ipynb).

</details>

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
Sample notebook [01.introduction.ipynb](https://github.com/taikinman/langrila/blob/main/notebooks/01.introduction.ipynb) includes following contents:

- Basic usage with simple text prompt
    - Chat Completion of OpenAI
    - Chat Completion on Azure OpenAI
    - Gemini of Google AI Studio
    - Gemini on VertexAI
    - Claude of Anthropic
    - Claude on Amazon Bedrock
- Universal message system in langrila
- How to specify system instruction
- Token management
- Usage gathering across multiple models
- Prompt template

[02.function_calling.ipynb](https://github.com/taikinman/langrila/blob/main/notebooks/02.function_calling.ipynb) instruct function calling in langrila.

- Basic usage for OpenAI Chat Completion, Gemini and Claude
- Universal tool config system
- Limitation for the function calling

[03.structured_output.ipynb](https://github.com/taikinman/langrila/blob/main/notebooks/03.structured_output.ipynb), you can see:

- JSON mode output for OpenAI and Gemini
- Pydantic schema output for OpenAI

[04.media_and_file_input.ipynb](https://github.com/taikinman/langrila/blob/main/notebooks/04.media_and_file_input.ipynb) show you the following contents:

- Image input
- PDF file input
- Video input
- Data uploading and analyzing by specifying uri for Gemini

[05.conversation_memory.ipynb](https://github.com/taikinman/langrila/blob/main/notebooks/05.conversation_memory.ipynb) provides you with how to store conversation history.

- Way to keep conversation history
- Multi-tuen conversation
- Multi-turn conversation with multiple client
- Multi-turn conversation using tools
- Multi-turn conversation using tools with multiple client
- Introduction conversation memory modules: 
    - JSONConversationMemory
    - PickleConversationMemory
    - CosmosConversationMemory for Azure Cosmos DB (Thanks to [@rioriost](https://github.com/rioriost))
    - S3ConversationMemory for AWS S3 (Thanks to [@kun432](https://github.com/kun432))

[06.embedding_text.ipynb](https://github.com/taikinman/langrila/blob/main/notebooks/06.embedding_text.ipynb)

- For OpenAI
- For Azure OpenAI
- For Gemini on Google AI Studio
- For Gemini on VertexAI

[07.basic_rag.ipynb](https://github.com/taikinman/langrila/blob/main/notebooks/07.basic_rag.ipynb)

- For Qdrant
- For Chroma
- For Usearch


# Dependencies
## must
- Python >=3.10,<3.13

## as needed
Langrila has various extra installation options. See the following installation section and [pyproject.toml](https://github.com/taikinman/langrila/blob/main/pyproject.toml).

# Installation
See extra dependencies section in [pyproject.toml](https://github.com/taikinman/langrila/blob/main/pyproject.toml) for more detail installation options.

## For user
### pip
```
# For OpenAI
pip install langrila[openai]

# For Gemini
pip install langrila[gemini]

# For Claude
pip install langrila[claude]

# For multiple clients
pip install langrila[openai,gemini,claude]

# With dependencies to handle specific data. Here is an example using gemini
pip install langrila[gemini,audio,video,pdf]

# With dependencies for specific platform. Here is an example using gemini on VertexAI
pip install langrila[gemini,vertexai]

# With dependencies for specific vectorDB. Here is an example using Qdrant
pip install langrila[openai,qdrant]
```

### poetry
```
# For OpenAI
poetry add langrila --extras openai

# For Gemini
poetry add langrila --extras gemini

# For Claude
poetry add langrila --extras claude

# For multiple clients
poetry add langrila --extras "openai gemini claude"

# With dependencies to handle specific data. Here is an example using gemini
poetry add langrila --extras "gemini audio video pdf"

# With dependencies for specific platform. Here is an example using gemini on VertexAI
poetry add langrila --extras "gemini vertexai"

# With dependencies for specific vectorDB. Here is an example using Qdrant
poetry add langrila --extras "openai qdrant"
```

## For developer
### clone
```
git clone git@github.com:taikinman/langrila.git
```

### pip
```
cd langrila

pip install -e .{extra packages}
```

### poetry
```
# For OpenAI
poetry add --editable /path/to/langrila/ --extras "{extra packages}"
```
