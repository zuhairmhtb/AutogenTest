import os
from typing import Any
from enum import Enum
from types import SimpleNamespace

from autogen_core.models import UserMessage
from autogen_ext.models.semantic_kernel import SKChatCompletionAdapter
from semantic_kernel import Kernel

from semantic_kernel.connectors.ai.hugging_face import HuggingFaceTextCompletion, HuggingFacePromptExecutionSettings
from semantic_kernel.memory.null_memory import NullMemory

from langchain_huggingface import HuggingFacePipeline
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

import torch


class HuggingFaceModelType(Enum):
    TEXT_COMPLETION = "text-completion"
    TEXT_TO_TEXT_GENERATION = "text2text-generation"
    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "question-answering"


class HuggingFaceQuestionAnsweringModel:
    def __init__(self, 
                 model_name:str, 
                 model_dir:str=None, 
                 device:int=-1, 
                 model_kwargs: dict[str, Any] | None = None,
                 pipeline_kwargs: dict[str, Any] | None = None,
                 temperature: float = 0.7):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
        self.model_dir = model_dir
        self.model_kwargs = model_kwargs
        self.pipeline_kwargs = pipeline_kwargs
        self.temperature = temperature

        # Move model to GPU if available
        self.device = device
        self.model.to(self.device)

    def _to_chatml_format(self, message:dict):
        """Convert message to ChatML format."""
        if message["role"] == "system":
            return SystemMessage(content=message["content"])
        if message["role"] == "assistant":
            return AIMessage(content=message["content"])
        if message["role"] == "user":
            return HumanMessage(content=message["content"])
        raise ValueError(f"Unknown message type: {type(message)}")

    def create(self, params):
        """Create a response using the model."""
        if params.get("stream", False) and "messages" in params:
            raise NotImplementedError("Local models do not support streaming.")

        num_of_responses = params.get("n", 1)
        response = SimpleNamespace()
        inputs = [self._to_chatml_format(m) for m in params["messages"]]
        response.choices = []
        response.model = self.model_name

        for _ in range(num_of_responses):
            outputs = self.model.invoke(inputs)
            text = outputs.content
            choice = SimpleNamespace()
            choice.message = SimpleNamespace()
            choice.message.content = text
            choice.message.function_call = None
            response.choices.append(choice)

        return response

    def message_retrieval(self, response):
        """Retrieve messages from the response."""
        return [choice.message.content for choice in response.choices]

    def cost(self, response) -> float:
        """Calculate the cost of the response."""
        response.cost = 0
        return 0

    @staticmethod
    def get_usage(response):
        """Get usage statistics."""
        return {}




class HuggingFaceModelFactory:
    """
    Factory class for creating SKChatCompletionAdapter with Hugging Face models
    Args:
        model_name (str): The name of the Hugging Face model
        model_type (HuggingFaceModelType, optional): The type of the model. Defaults to HuggingFaceModelType.TEXT_TO_TEXT_GENERATION.
        model_dir (str, optional): The directory where the model is stored. Defaults to None.
        device (int, optional): The device to run the model on. Defaults to -1.
        model_kwargs (dict[str, Any], optional): Additional model kwargs. Defaults to None.
        pipeline_kwargs (dict[str, Any], optional): Additional pipeline kwargs. Defaults to None.
        temperature (float, optional): The temperature for the model. Defaults to 0.7.
    Returns:
        SKChatCompletionAdapter: The adapter
    """
    def __init__(self, 
                 model_name: str, 
                 model_type:HuggingFaceModelType="text2text-generation", 
                 model_dir:str = None,
                 device:int=-1,
                 model_kwargs: dict[str, Any] | None = None,
                 pipeline_kwargs: dict[str, Any] | None = None,
                 temperature: float = 0.7,):
        self.model_name = model_name
        self.model_type = model_type
        self.model_dir = model_dir
        self.device = device
        self.model_kwargs = model_kwargs
        self.pipeline_kwargs = pipeline_kwargs
        self.temperature = temperature

        if not self._is_model_supported():
            raise ValueError(f"Model {self.model_name} is not supported")


    def _is_model_supported(self)->bool:
        """
        Check if the model is supported
        Returns:
            bool: True if the model is supported, False otherwise
        """
        try:
            semantic_client = HuggingFaceTextCompletion(
                ai_model_id=self.model_name,
                task=self.model_type,
                device=self.device,
                model_kwargs=self.model_kwargs,
                pipeline_kwargs=self.pipeline_kwargs
            )
            settings = HuggingFacePromptExecutionSettings(
                temperature=self.temperature
            )
            client = SKChatCompletionAdapter(
                sk_client=semantic_client,
                kernel=Kernel(memory=NullMemory()),
                prompt_settings=settings
            )
            client.create(
                UserMessage(
                    source="user",
                    content="What dataset was TinyRoBERTa trained on?"
                )
            )
            return True
        except:
            return False
    
    def _build_qa_adapter(self)->SKChatCompletionAdapter:
        """
        """
        pass
    def build(self)->SKChatCompletionAdapter:
        """
        Build a SKChatCompletionAdapter with the specified model
        Returns:
            SKChatCompletionAdapter: The adapter
        """
        if self.model_type == HuggingFaceModelType.QUESTION_ANSWERING:
            return self._build_qa_adapter()
        semantic_client = HuggingFaceTextCompletion(
                ai_model_id=self.model_name,
                task=self.model_type,
                device=self.device,
                model_kwargs=self.model_kwargs,
                pipeline_kwargs=self.pipeline_kwargs
         )
        settings = HuggingFacePromptExecutionSettings(
            temperature=self.temperature
        )
        return SKChatCompletionAdapter(
            sk_client=semantic_client,
            kernel=Kernel(memory=NullMemory()),
            prompt_settings=settings
        )