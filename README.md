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

# Supported models for OpenAI
## Chat models
- gpt-3.5-turbo-1106
- gpt-3.5-turbo-0125
- gpt-4-1106-preview
- gpt-4-vision-preview
- gpt-4-0125-preview
- gpt-4-turbo-2024-04-09
- gpt-4o-2024-05-13
- gpt-4o-mini-2024-07-18
- gpt-4o-2024-08-06
- chatgpt-4o-latest

## Embedding models
- text-embedding-ada-002
- text-embedding-3-small
- text-embedding-3-large

## Aliases
```
{'gpt-4o-mini': 'gpt-4o-mini-2024-07-18',
 'gpt-4o': 'gpt-4o-2024-08-06',
 'gpt-4-turbo': 'gpt-4-turbo-2024-04-09',
 'gpt-3.5-turbo': 'gpt-3.5-turbo-0125'}
```

## Platform
- OpenAI
- Azure OpenAI

# Supported models for Gemini
## Chat models
Basically all the models of gemini-1.5 family including experimental models.

## Platform
- Google AI
- VertexAI

# Supported models for Claude
## Chat models
- claude-3.5-sonnet
- claude-3-opus
- claude-3-sonnet
- claude-3-haiku

## Platform
- Anthropic
- Amazon Bedrock
- VertexAI (not tested)

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
    - Gemini of Google AI
    - Gemini on VertexAI
    - Claude of Anthropic
    - Claude on Amazon Bedrock
- Message system in langrila
- Multi-turn conversation with multiple client
- How to specify system instruction
- Token management
- Usage gathering across multiple models
- Prompt template

[02.function_calling.ipynb](https://github.com/taikinman/langrila/blob/main/notebooks/02.function_calling.ipynb) instruct function calling in langrila.

- Basic usage for OpenAI Chat Completion, Gemini and Claude
- Multi-turn conversation using tools
- Multi-turn conversation using tools with multiple client

[03.structured_output.ipynb](https://github.com/taikinman/langrila/blob/main/notebooks/03.structured_output.ipynb), you can see:

- JSON mode output for OpenAI and Gemini
- Pydantic schema output for OpenAI

[04.media_and_file_input.ipynb](https://github.com/taikinman/langrila/blob/main/notebooks/04.media_and_file_input.ipynb) show you the following contents:

- Image input
- PDF file input
- Video input
- Data uploading and analyzing by specifying uri for Gemini

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

# Optional
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

For more details, see [qdrant.py](https://github.com/taikinman/langrila/blob/main/src/langrila/database/qdrant.py).

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

For more details, see [chroma.py](https://github.com/taikinman/langrila/blob/main/src/langrila/database/chroma.py).

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

When you need to filter retrieval results by metadata in search time, you can implement your custom metadata filter. Base class of metadata filter is in [base.py](https://github.com/taikinman/langrila/blob/main/src/langrila/base.py). For more details, see : [usearch.py](https://github.com/taikinman/langrila/blob/main/src/langrila/database/usearch.py).

### Specific use case
The library supports a variety of use cases by combining modules such as these and defining new modules. For example, the following is an example of a module that combines basic Retrieval and prompt templates. 
