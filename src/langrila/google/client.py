import json
import os
from typing import Any, AsyncGenerator, Generator, Literal, Sequence, cast

from google import genai  # type: ignore
from google.auth.credentials import Credentials
from google.genai.types import (
    Blob,
    Content,
    EmbedContentConfig,
    FileData,
    FunctionCall,
    FunctionDeclaration,
    FunctionResponse,
    GenerateContentConfig,
    GenerateImageConfig,
    Part,
)
from google.genai.types import Tool as GeminiTool

from ..core.client import LLMClient
from ..core.embedding import EmbeddingResults
from ..core.prompt import (
    AudioPrompt,
    ImagePrompt,
    PDFPrompt,
    Prompt,
    SystemPrompt,
    TextPrompt,
    ToolCallPrompt,
    ToolUsePrompt,
    URIPrompt,
    VideoPrompt,
)
from ..core.response import (
    ImageResponse,
    Response,
    ResponseType,
    TextResponse,
    ToolCallResponse,
)
from ..core.tool import Tool
from ..core.usage import Usage
from ..utils import (
    create_parameters,
    generate_dummy_call_id,
    make_batch,
    snake_to_camel,
    utf8_to_bytes,
)
from .gemini_utils import recurse_transform_type_to_upper

GeminiMessage = Content | str


class GoogleClient(LLMClient[Content, str, Part, GeminiTool]):
    """
    Wrapper client for interacting with the Gemini API.

    Parameters
    ----------
    api_key_env_name : str, optional
        Name of the environment variable containing the API key.
    api_type : str, optional
        Type of API to use. Either "aistudio" or "vertexai".
    project_id_env_name : str, optional
        Name of the environment variable containing the project ID.
    location : str, optional
        Location of the API.
    credentials : Credentials, optional
        Google credentials to use.
    """

    def __init__(
        self,
        api_key_env_name: str | None = None,
        api_type: Literal["aistudio", "vertexai"] = "aistudio",
        project_id_env_name: str | None = None,
        location: str | None = None,
        credentials: Credentials | None = None,
    ):
        self.api_key = os.getenv(api_key_env_name) if api_key_env_name else None
        self.project_id = os.getenv(project_id_env_name) if project_id_env_name else None
        self.location = location
        self.credentials = credentials
        self.vertexai = api_type == "vertexai"
        self.client = genai.Client(
            vertexai=self.vertexai,
            api_key=self.api_key,
            project=self.project_id,
            location=self.location,
            credentials=self.credentials,
        )

    def setup_system_instruction(self, system_instruction: SystemPrompt) -> str:
        """
        Setup the system instruction.

        Parameters
        ----------
        system_instruction : SystemPrompt
            System instruction.

        Returns
        ----------
        str
            List of messages with the system instruction.
        """
        return system_instruction.contents

    def generate_text(
        self, messages: list[Content], system_instruction: str | None = None, **kwargs: Any
    ) -> Response:
        """
        Generate text from a list of messages.

        Parameters
        ----------
        messages : list[Content]
            List of messages to generate text from.
        system_instruction : str, optional
            System instruction to include in the messages.
        **kwargs : Any
            Additional keyword arguments.
            Basically the same as the parameters in `google.genai.types.GenerateContentConfig`.
            For more details, see the document of the `google-genai` package.

        Returns
        ----------
        Response
            Response object containing generated text.
        """
        if system_instruction:
            kwargs["system_instruction"] = system_instruction

        _kwargs = {snake_to_camel(k): v for k, v in kwargs.items()}
        generation_config_params = create_parameters(GenerateContentConfig, None, None, **_kwargs)
        generation_config = GenerateContentConfig(**generation_config_params)

        response = self.client.models.generate_content(
            model=kwargs.get("model"),
            contents=messages,
            config=generation_config,
        )

        contents: list[TextResponse | ToolCallResponse] = []
        for candidate in response.candidates:
            if candidate.content:
                for part in candidate.content.parts:
                    if part.text:
                        if text := part.text.strip():
                            contents.append(TextResponse(text=text))
                    elif part.function_call:
                        call_id = part.function_call.id or generate_dummy_call_id(n=24)
                        contents.append(
                            ToolCallResponse(
                                call_id=call_id,
                                name=part.function_call.name,
                                args=json.dumps(part.function_call.args),
                            )
                        )

        return Response(
            contents=cast(list[ResponseType], contents),
            usage=Usage(
                model_name=cast(str | None, kwargs.get("model")),
                prompt_tokens=response.usage_metadata.prompt_token_count or 0,
                output_tokens=response.usage_metadata.candidates_token_count or 0,
            ),
            raw=response,
            name=cast(str | None, kwargs.get("name")),
        )

    async def generate_text_async(
        self, messages: list[Content], system_instruction: str | None = None, **kwargs: Any
    ) -> Response:
        """
        Generate text from a list of messages asynchronously.

        Parameters
        ----------
        messages : list[Content]
            List of messages to generate text from.
        system_instruction : str, optional
            System instruction.
        **kwargs : Any
            Additional keyword arguments.
            Basically the same as the parameters in `google.genai.types.GenerateContentConfig`.
            For more details, see the document of the `google-genai` package.

        Returns
        ----------
        Response
            Response object containing generated text.
        """
        if system_instruction:
            kwargs["system_instruction"] = system_instruction

        _kwargs = {snake_to_camel(k): v for k, v in kwargs.items()}
        generation_config_params = create_parameters(GenerateContentConfig, None, None, **_kwargs)
        generation_config = GenerateContentConfig(**generation_config_params)

        response = await self.client.aio.models.generate_content(
            model=kwargs.get("model"),
            contents=messages,
            config=generation_config,
        )

        contents: list[TextResponse | ToolCallResponse] = []
        for candidate in response.candidates:
            if candidate.content:
                for part in candidate.content.parts:
                    if part.text:
                        if text := part.text.strip():
                            contents.append(TextResponse(text=text))
                    elif part.function_call:
                        call_id = part.function_call.id or generate_dummy_call_id(n=24)
                        contents.append(
                            ToolCallResponse(
                                call_id=call_id,
                                name=part.function_call.name,
                                args=json.dumps(part.function_call.args),
                            )
                        )

        return Response(
            contents=cast(list[ResponseType], contents),
            usage=Usage(
                model_name=cast(str | None, kwargs.get("model")),
                prompt_tokens=response.usage_metadata.prompt_token_count or 0,
                output_tokens=response.usage_metadata.candidates_token_count or 0,
            ),
            raw=response,
            name=cast(str | None, kwargs.get("name")),
        )

    def stream_text(
        self, messages: list[Content], system_instruction: str | None = None, **kwargs: Any
    ) -> Generator[Response, None, None]:
        """
        Stream text from a list of messages.

        Parameters
        ----------
        messages : list[Content]
            List of messages to stream text from.
        system_instruction : str, optional
            System instruction.
        **kwargs : Any
            Additional keyword arguments.
            Basically the same as the parameters in `google.genai.types.GenerateContentConfig`.
            For more details, see the document of the `google-genai` package.

        Yields
        ----------
        Response
            Response object containing combined chunk text or args.
        """
        if system_instruction:
            kwargs["system_instruction"] = system_instruction

        _kwargs = {snake_to_camel(k): v for k, v in kwargs.items()}
        generation_config_params = create_parameters(GenerateContentConfig, None, None, **_kwargs)
        generation_config = GenerateContentConfig(**generation_config_params)

        streamed_response = self.client.models.generate_content_stream(
            model=kwargs.get("model"),
            contents=messages,
            config=generation_config,
        )

        chunk_texts = ""
        usage = Usage(model_name=cast(str | None, kwargs.get("model")))
        res: TextResponse | ToolCallResponse | None = None
        contents: list[TextResponse | ToolCallResponse] = []
        for chunk in streamed_response:
            if usage_metadata := chunk.usage_metadata:
                usage = Usage(
                    model_name=cast(str | None, kwargs.get("model")),
                    prompt_tokens=usage_metadata.prompt_token_count or 0,
                    output_tokens=usage_metadata.candidates_token_count or 0,
                )

            for candidate in chunk.candidates:
                for part in candidate.content.parts:
                    if part.text and part.text.strip():
                        chunk_texts += part.text
                        res = TextResponse(text=chunk_texts)

                        yield Response(
                            contents=[res],
                            usage=usage,
                            raw=chunk,
                            name=cast(str | None, kwargs.get("name")),
                        )
                        continue
                    elif part.text and not part.text.strip():
                        continue

                    if res:
                        contents.append(res)

                    chunk_texts = ""

                    if part.function_call:  # non-streaming
                        call_id = part.function_call.id or generate_dummy_call_id(n=24)
                        args = json.dumps(part.function_call.args)
                        res = ToolCallResponse(
                            call_id=call_id,
                            name=part.function_call.name,
                            args=args,
                        )
                        yield Response(
                            contents=[res],
                            usage=usage,
                            raw=chunk,
                            name=cast(str | None, kwargs.get("name")),
                        )
                    else:
                        continue

                    if res:
                        contents.append(res)

                    res = None

        if contents:
            yield Response(
                contents=cast(list[ResponseType], contents),
                usage=usage,
                raw=chunk,
                name=cast(str | None, kwargs.get("name")),
                is_last_chunk=True,
            )

    async def stream_text_async(
        self, messages: list[Content], system_instruction: str | None = None, **kwargs: Any
    ) -> AsyncGenerator[Response, None]:
        """
        Stream text from a list of messages asynchronously.

        Parameters
        ----------
        messages : list[Content]
            List of messages to stream text from.
        system_instruction : str, optional
            System instruction.
        **kwargs : Any
            Additional keyword arguments.
            Basically the same as the parameters in `google.genai.types.GenerateContentConfig`.
            For more details, see the document of the `google-genai` package.

        Yields
        ----------
        Response
            Response object containing combined chunk text or args.
        """
        if system_instruction:
            kwargs["system_instruction"] = system_instruction

        _kwargs = {snake_to_camel(k): v for k, v in kwargs.items()}
        generation_config_params = create_parameters(GenerateContentConfig, None, None, **_kwargs)
        generation_config = GenerateContentConfig(**generation_config_params)

        streamed_response = self.client.aio.models.generate_content_stream(
            model=kwargs.get("model"),
            contents=messages,
            config=generation_config,
        )

        chunk_texts = ""
        usage = Usage(model_name=cast(str | None, kwargs.get("model")))
        res: TextResponse | ToolCallResponse | None = None
        contents: list[TextResponse | ToolCallResponse] = []
        async for chunk in streamed_response:
            if usage_metadata := chunk.usage_metadata:
                usage = Usage(
                    model_name=cast(str | None, kwargs.get("model")),
                    prompt_tokens=usage_metadata.prompt_token_count or 0,
                    output_tokens=usage_metadata.candidates_token_count or 0,
                )

            for candidate in chunk.candidates:
                for part in candidate.content.parts:
                    if part.text and part.text.strip():
                        chunk_texts += part.text
                        res = TextResponse(text=chunk_texts)

                        yield Response(
                            contents=[res],
                            usage=usage,
                            raw=chunk,
                            name=cast(str | None, kwargs.get("name")),
                        )
                        continue
                    elif part.text and not part.text.strip():
                        continue

                    if res:
                        contents.append(res)

                    chunk_texts = ""

                    if part.function_call:  # non-streaming
                        call_id = part.function_call.id or generate_dummy_call_id(n=24)
                        args = json.dumps(part.function_call.args)
                        res = ToolCallResponse(
                            call_id=call_id,
                            name=part.function_call.name,
                            args=args,
                        )
                        yield Response(
                            contents=[res],
                            usage=usage,
                            raw=chunk,
                            name=cast(str | None, kwargs.get("name")),
                        )
                    else:
                        continue

                    if res:
                        contents.append(res)

                    res = None

        if contents:
            yield Response(
                contents=cast(list[ResponseType], contents),
                usage=usage,
                raw=chunk,
                name=cast(str | None, kwargs.get("name")),
                is_last_chunk=True,
            )

    def embed_text(self, texts: Sequence[str], **kwargs: Any) -> EmbeddingResults:
        """
        Embed text into a vector space.

        Parameters
        ----------
        texts : Sequence[str]
            Texts to embed.
        **kwargs : Any
            Additional keyword arguments.
            Basically the same as the parameters in `google.genai.types.EmbedContentConfig`.
            For more details, see the document of the `google-genai` package.

        Returns
        ----------
        EmbeddingResults
            Embedding results.
        """
        if not (isinstance(texts, list) or isinstance(texts, str)):
            raise ValueError("Texts must be a string, not a list of strings.")

        if not isinstance(texts, list):
            texts = [texts]

        _kwargs = {snake_to_camel(k): v for k, v in kwargs.items()}
        embed_config_params = create_parameters(EmbedContentConfig, None, None, **_kwargs)
        embed_config = EmbedContentConfig(**embed_config_params)

        total_usage = Usage(model_name=kwargs.get("model"))
        batch_size = kwargs.get("batch_size", 10)
        embeddings = []
        for batch in make_batch(texts, batch_size=batch_size):
            embedding = self.client.models.embed_content(
                model=kwargs.get("model"),
                contents=batch,
                config=embed_config,
            )
            embeddings.extend([emb.values for emb in embedding.embeddings])
            if matadata := embedding.metadata:
                if billable_character_count := matadata.billable_character_count:
                    total_usage += Usage(prompt_tokens=billable_character_count or 0)

        return EmbeddingResults(
            text=texts,
            embeddings=embeddings,
            usage=total_usage,
        )

    async def embed_text_async(self, texts: Sequence[str], **kwargs: Any) -> EmbeddingResults:
        """
        Embed text into a vector space asynchronously.

        Parameters
        ----------
        texts : Sequence[str]
            Texts to embed.
        **kwargs : Any
            Additional keyword arguments.
            Basically the same as the parameters in `google.genai.types.EmbedContentConfig`.
            For more details, see the document of the `google-genai` package.

        Returns
        ----------
        EmbeddingResults
            Embedding results.
        """
        if not (isinstance(texts, list) or isinstance(texts, str)):
            raise ValueError("Texts must be a string, not a list of strings.")

        if not isinstance(texts, list):
            texts = [texts]

        _kwargs = {snake_to_camel(k): v for k, v in kwargs.items()}
        embed_config_params = create_parameters(EmbedContentConfig, None, None, **_kwargs)
        embed_config = EmbedContentConfig(**embed_config_params)

        total_usage = Usage(model_name=kwargs.get("model"))
        batch_size = kwargs.get("batch_size", 10)
        embeddings = []
        for batch in make_batch(texts, batch_size=batch_size):
            embedding = await self.client.aio.models.embed_content(
                model=kwargs.get("model"),
                contents=batch,
                config=embed_config,
            )
            embeddings.extend([emb.values for emb in embedding.embeddings])
            if matadata := embedding.metadata:
                if billable_character_count := matadata.billable_character_count:
                    total_usage += Usage(prompt_tokens=billable_character_count or 0)

        return EmbeddingResults(
            text=texts,
            embeddings=embeddings,
            usage=total_usage,
        )

    def generate_image(self, prompt: str, **kwargs: Any) -> Response:
        """
        Generate an image from a prompt.

        Parameters
        ----------
        prompt : str
            Prompt to generate an image from.
        **kwargs : Any
            Additional keyword arguments.
            Basically the same as the parameters in `google.genai.types.GenerateImageConfig`.
            For more details, see the document of the `google-genai` package.

        Returns
        ----------
        Response
            Response object containing generated image.
        """
        _kwargs = {snake_to_camel(k): v for k, v in kwargs.items()}
        embed_config_params = create_parameters(GenerateImageConfig, None, None, **_kwargs)
        embed_config = GenerateImageConfig(**embed_config_params)

        image = self.client.models.generate_image(
            model=kwargs.get("model"),
            prompt=prompt,
            config=embed_config,
        )

        contents = []
        for generated_image in image.generated_images:
            pil_image = generated_image.image._pil_image
            img_format = pil_image.format.lower() if pil_image.format else "png"
            contents.append(ImageResponse(image=pil_image, format=img_format))

        return Response(
            contents=contents,
            usage=Usage(),
            raw=image,
            name=cast(str | None, kwargs.get("name")),
        )

    async def generate_image_async(self, prompt: str, **kwargs: Any) -> Response:
        """
        Generate an image from a prompt asynchronously.

        Parameters
        ----------
        prompt : str
            Prompt to generate an image from.
        **kwargs : Any
            Additional keyword arguments.
            Basically the same as the parameters in `google.genai.types.GenerateImageConfig`.
            For more details, see the document of the `google-genai` package.

        Returns
        ----------
        Response
            Response object containing generated image.
        """
        _kwargs = {snake_to_camel(k): v for k, v in kwargs.items()}
        embed_config_params = create_parameters(GenerateImageConfig, None, None, **_kwargs)
        embed_config = GenerateImageConfig(**embed_config_params)

        image = await self.client.aio.models.generate_image(
            model=kwargs.get("model"),
            prompt=prompt,
            config=embed_config,
        )

        contents = []
        for generated_image in image.generated_images:
            pil_image = generated_image.image._pil_image
            img_format = pil_image.format.lower() if pil_image.format else "png"
            contents.append(ImageResponse(image=pil_image, format=img_format))

        return Response(
            contents=contents,
            usage=Usage(),
            raw=image,
            name=cast(str | None, kwargs.get("name")),
        )

    def map_to_client_prompt(self, message: Prompt) -> Content:
        """
        Map a message to a client-specific representation.

        Parameters
        ----------
        message : Prompt
            Prompt to map.

        Returns
        ----------
        Content
            Client-specific message representation.
        """
        parts: list[Part] = []
        for content in message.contents:
            if isinstance(content, str):
                parts.append(Part(text=content))
            elif isinstance(content, TextPrompt):
                parts.append(Part(text=content.text))
            elif isinstance(content, ImagePrompt):
                parts.append(
                    Part(
                        inline_data=Blob(
                            data=utf8_to_bytes(cast(str, content.image)),
                            mime_type=f"image/{content.format}",
                        )
                    )
                )
            elif isinstance(content, PDFPrompt):
                parts.extend(
                    [
                        Part(
                            inline_data=Blob(
                                data=utf8_to_bytes(cast(str, img.image)),
                                mime_type="image/jpeg",
                            )
                        )
                        for img in content.as_image_content()
                    ]
                )
            elif isinstance(content, URIPrompt):
                parts.append(
                    Part(file_data=FileData(file_uri=content.uri, mime_type=content.mime_type))
                )
            elif isinstance(content, VideoPrompt):
                parts.extend(
                    [
                        Part(
                            inline_data=Blob(
                                data=utf8_to_bytes(cast(str, img.image)),
                                mime_type="image/jpeg",
                            )
                        )
                        for img in content.as_image_content()
                    ]
                )
            elif isinstance(content, AudioPrompt):
                parts.append(
                    Part(
                        inline_data=Blob(
                            data=utf8_to_bytes(cast(str, content.audio)),
                            mime_type=content.mime_type,
                        )
                    )
                )
            elif isinstance(content, ToolCallPrompt):
                content_args = content.args
                if isinstance(content_args, str) and content_args:
                    content_args = json.loads(content_args)

                parts.append(
                    Part(
                        function_call=FunctionCall(
                            id=content.call_id
                            if not self.vertexai
                            else None,  # VertexAI does not support id
                            args=content_args,
                            name=content.name,
                        )
                    )
                )
            elif isinstance(content, ToolUsePrompt):
                if content.output:
                    parts.append(
                        Part(
                            function_response=FunctionResponse(
                                id=content.call_id
                                if not self.vertexai
                                else None,  # VertexAI does not support id
                                name=content.name,
                                response={"output": content.output},
                            )
                        )
                    )
                elif content.error:
                    parts.append(
                        Part(
                            function_response=FunctionResponse(
                                id=content.call_id
                                if not self.vertexai
                                else None,  # VertexAI does not support id
                                name=content.name,
                                response={"error": content.error},
                            )
                        )
                    )

        # return parts
        return Content(role=message.role, parts=parts)

    def map_to_client_tools(self, tools: list[Tool], **kwargs: Any) -> list[GeminiTool]:
        return [
            GeminiTool(
                function_declarations=[self.map_to_client_tool(tool=tool) for tool in tools],
                **kwargs,
            )
        ]

    def map_to_client_tool(self, tool: Tool, **kwargs: Any) -> FunctionDeclaration:
        if tool.schema_dict is None:
            raise ValueError("Tool schema is required.")

        schema = recurse_transform_type_to_upper(tool.schema_dict)
        return FunctionDeclaration(
            description=tool.description,
            parameters=schema
            if schema.get("properties")
            else None,  # Google Developer API does not support empty properties
            response=kwargs.get("response"),
            name=tool.name,
        )
