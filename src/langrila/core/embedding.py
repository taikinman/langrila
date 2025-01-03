from .pydantic import BaseModel
from .usage import Usage


class EmbeddingResults(BaseModel):
    text: list[str]
    embeddings: list[list[float]]
    usage: Usage
