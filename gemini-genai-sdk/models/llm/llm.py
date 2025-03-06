import base64
import tempfile
import json
import time
import os
from collections.abc import Generator
from typing import Optional, Union

import requests
from google import genai
from google.genai import types
from google.genai import errors

from dify_plugin.entities.model.llm import LLMResult, LLMResultChunk, LLMResultChunkDelta
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    PromptMessage,
    PromptMessageContent,
    PromptMessageContentType,
    PromptMessageTool,
    SystemPromptMessage,
    ToolPromptMessage,
    UserPromptMessage,
)
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeAuthorizationError,
    InvokeBadRequestError,
    InvokeConnectionError,
    InvokeError,
    InvokeRateLimitError,
    InvokeServerUnavailableError,
)
from dify_plugin.interfaces.model.large_language_model import LargeLanguageModel

class GoogleLargeLanguageModel(LargeLanguageModel):
    def _invoke(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        """
        Invoke large language model

        :param model: model name
        :param credentials: model credentials
        :param prompt_messages: prompt messages
        :param model_parameters: model parameters
        :param tools: tools for tool calling
        :param stop: stop words
        :param stream: is stream response
        :param user: unique user id
        :return: full response or stream response chunk generator result
        """
        return self._generate(model, credentials, prompt_messages, model_parameters, tools, stop, stream, user)

    def get_num_tokens(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        tools: Optional[list[PromptMessageTool]] = None,
    ) -> int:
        # FIXME: implement get_num_tokens
        return 0

    def _convert_tools_to_genai_tool(self, tools: list[PromptMessageTool]) -> types.Tool:
        """
        Convert tool messages to tools

        :param tools: tool messages
        :return: types.Tool
        """
        function_declarations = []
        for tool in tools:
            properties = {}
            for key, value in tool.parameters.get("properties", {}).items():
                properties[key] = {
                    "type_": types.Type.STRING,
                    "description": value.get("description", ""),
                    "enum": value.get("enum", []),
                }
            if properties:
                parameters = types.Schema(
                    type=types.Type.OBJECT, properties=properties, required=tool.parameters.get("required", [])
                )
            else:
                parameters = None

            function_declarations.append(
                types.FunctionDeclaration(
                    description=tool.description,
                    name=tool.name,
                    parameters=parameters,
                )
            )
        return types.Tool(function_declarations=function_declarations)

    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        Validate model credentials

        :param model: model name
        :param credentials: model credentials
        :return:
        """
        try:
            ping_message = UserPromptMessage(content="ping")
            self._generate(model, credentials, [ping_message], stream=False, model_parameters={"max_output_tokens": 5})
        except Exception as ex:
            raise CredentialsValidateFailedError(str(ex))

    def _generate(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        """
        Invoke large language model

        :param model: model name
        :param credentials: credentials kwargs
        :param prompt_messages: prompt messages
        :param model_parameters: model parameters
        :param stop: stop words
        :param stream: is stream response
        :param user: unique user id
        :return: full response or stream response chunk generator result
        """
        config_kwargs = model_parameters.copy()

        config = types.GenerateContentConfig()
        genai_tools = []

        if tools:
            genai_tools.extend(self._convert_tools_to_genai_tool(tools))

        if config_kwargs.pop("grounding", False):
            genai_tools.append(
                types.Tool(google_search=types.GoogleSearchRetrieval)
            )

        if response_format := config_kwargs.pop("response_format", None):
            if response_format == "json_object":
                config.response_mime_type = "application/json"

        if safety_settings := config_kwargs.pop("safety_settings", None):
            config.safety_settings = [
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=types.HarmBlockThreshold[safety_settings],
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=types.HarmBlockThreshold[safety_settings],
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=types.HarmBlockThreshold[safety_settings],
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold[safety_settings],
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                    threshold=types.HarmBlockThreshold[safety_settings],
                ),
            ]

        if len(genai_tools) > 0:
            config.tools = genai_tools

        for key, value in config_kwargs.items():
            config.__setattr__(key, value)

        history = []
        for msg in prompt_messages:
            content = self._format_message_to_content(msg)
            if content["role"] == "system":
                if config.system_instruction:
                    config.system_instruction = [*config.system_instruction, content["parts"]]
                else:
                    config.system_instruction = content["parts"]
            else:
                history.append(types.Content(parts=content["parts"], role=content["role"]))

        if not history:
            raise InvokeError("The user prompt message is required. You only add a system prompt message.")

        client = genai.Client(
            api_key=credentials["google_api_key"],
            http_options=types.HttpOptions(
                base_url=credentials["endpoint_url"] or "https://generativelanguage.googleapis.com",
            )
        )
        if stream:
            response = client.models.generate_content_stream(
                model=model,
                contents=history,
                config=config,
            )

            return self._handle_generate_stream_response(model, credentials, response, prompt_messages)
        else:
            response = client.models.generate_content(
                model=model,
                contents=history,
                config=config,
            )
            return self._handle_generate_response(model, credentials, response, prompt_messages)

    def _handle_generate_response(
        self, model: str, credentials: dict, response: types.GenerateContentResponse, prompt_messages: list[PromptMessage]
    ) -> LLMResult:
        """
        Handle llm response

        :param model: model name
        :param credentials: credentials
        :param response: response
        :param prompt_messages: prompt messages
        :return: llm response
        """
        assistant_prompt_message = AssistantPromptMessage(content=response.text)
        if response.usage_metadata:
            prompt_tokens = response.usage_metadata.prompt_token_count
            completion_tokens = response.usage_metadata.candidates_token_count
        else:
            prompt_tokens = self.get_num_tokens(model, credentials, prompt_messages)
            completion_tokens = self.get_num_tokens(model, credentials, [assistant_prompt_message])
        usage = self._calc_response_usage(model, credentials, prompt_tokens, completion_tokens)
        result = LLMResult(model=model, prompt_messages=prompt_messages, message=assistant_prompt_message, usage=usage)
        return result

    def _handle_generate_stream_response(
        self, model: str, credentials: dict, response: types.GenerateContentResponse, prompt_messages: list[PromptMessage]
    ) -> Generator:
        """
        Handle llm stream response

        :param model: model name
        :param credentials: credentials
        :param response: response
        :param prompt_messages: prompt messages
        :return: llm response chunk generator result
        """
        index = -1
        for chunk in response:
            parts = chunk.candidates[0].content.parts
            for part in parts:
                assistant_prompt_message = AssistantPromptMessage(content="")
                if part.text:
                    assistant_prompt_message.content += part.text
                if part.function_call:
                    assistant_prompt_message.tool_calls = [
                        AssistantPromptMessage.ToolCall(
                            id=part.function_call.name,
                            type="function",
                            function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                                name=part.function_call.name,
                                arguments=json.dumps(dict(part.function_call.args.items())),
                            ),
                        )
                    ]
                index += 1
                finish_reason = str(chunk.candidates[0].finish_reason)
                if not finish_reason:
                    yield LLMResultChunk(
                        model=model,
                        prompt_messages=prompt_messages,
                        delta=LLMResultChunkDelta(index=index, message=assistant_prompt_message),
                    )
                else:
                    if hasattr(response, "usage_metadata") and response.usage_metadata:
                        prompt_tokens = response.usage_metadata.prompt_token_count
                        completion_tokens = response.usage_metadata.candidates_token_count
                    else:
                        prompt_tokens = self.get_num_tokens(model, credentials, prompt_messages)
                        completion_tokens = self.get_num_tokens(model, credentials, [assistant_prompt_message])
                    usage = self._calc_response_usage(model, credentials, prompt_tokens, completion_tokens)
                    yield LLMResultChunk(
                        model=model,
                        prompt_messages=prompt_messages,
                        delta=LLMResultChunkDelta(
                            index=index,
                            message=assistant_prompt_message,
                            finish_reason=finish_reason,
                            usage=usage,
                        ),
                    )

    def _format_message_to_content(self, message: PromptMessage) -> list:
        if isinstance(message, UserPromptMessage):
            content = {"role": "user", "parts": []}

            if isinstance(message.content, str):
                content["parts"].append(types.Part.from_text(text=message.content))
            else:
                for c in message.content:
                    if c.type == PromptMessageContentType.TEXT:
                        content["parts"].append(types.Part.from_text(text=c.data))
                    else:
                        # FIXME: upload file to google
                        content["parts"].append(types.Part.from_bytes(data=base64.b64decode(c.base64_data), mime_type=c.mime_type))
        elif isinstance(message, AssistantPromptMessage):
            content = {"role": "model", "parts": []}
            if message.content:
                content["parts"].append(types.Part.from_text(text=message.content))
            if message.tool_calls:
                content["parts"].append(types.Part.from_function_call(name=message.tool_calls[0].id))
        elif isinstance(message, SystemPromptMessage):
            content = {"role": "system", "parts": [types.Part.from_text(text=message.content)]}
        elif isinstance(message, ToolPromptMessage):
            content = {"role": "function", "parts": [types.Part.from_function_call(name=message.name)]}
        else:
            raise ValueError(f"Got unknown type {message}")

        return content

    @property
    def _invoke_error_mapping(self) -> dict[type[InvokeError], list[type[Exception]]]:
        return {
            InvokeServerUnavailableError: [
                errors.ServerError,
                errors.ExperimentalWarning
            ],
            InvokeBadRequestError: [
                errors.ClientError,
                errors.UnknownFunctionCallArgumentError,
                errors.UnsupportedFunctionError,
                errors.FunctionInvocationError,
            ],
        }
