import logging
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class ChatSFTDataset(Dataset):
    """
    Dataset for chat-style supervised fine-tuning data.

    Each sample expects a ``messages`` list following the OpenAI / Tulu chat
    schema. Tokenization relies on ``tokenizer.apply_chat_template`` to
    construct the concatenated conversation and masks non-assistant tokens
    (following AllenAI's Open-Instruct preprocessing).
    """

    def __init__(self,
                 samples: List[Dict[str, Any]],
                 tokenizer,
                 max_length: int = 4096):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.samples[index]
        messages = record.get("messages", [])
        if not isinstance(messages, list) or len(messages) == 0:
            raise ValueError("Each sample must include a non-empty messages list")

        encoding = self._tokenize_messages(messages)
        example = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "labels": encoding["labels"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if "task_family" in record:
            example["task_family"] = record["task_family"]
        if "source" in record:
            example["source"] = record["source"]
        return example

    def _tokenize_messages(self, messages: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        """
        Convert multi-turn messages into token ids and labels while masking
        non-assistant tokens (borrowed from Open-Instruct's SFT pipeline).
        """
        # Primary tokenization of the full conversation.
        if hasattr(self.tokenizer, "apply_chat_template"):
            input_ids = self.tokenizer.apply_chat_template(
                conversation=messages,
                tokenize=True,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.max_length,
                add_generation_prompt=False,
            )
        else:
            joined = self._convert_messages_to_text(messages)
            input_ids = self.tokenizer(
                joined,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )["input_ids"]
        labels = input_ids.clone()

        for idx, message in enumerate(messages):
            role = message.get("role")
            if role == "assistant":
                continue

            message_start = self._measure_prefix_length(messages[:idx])
            message_end = self._measure_prefix_length(
                messages[:idx + 1],
                add_generation_prompt=(idx < len(messages) - 1
                                       and messages[idx + 1].get("role") == "assistant"))

            labels[:, message_start:message_end] = -100
            if self.max_length and message_end >= self.max_length:
                break

        attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids[:, :self.max_length],
            "labels": labels[:, :self.max_length],
            "attention_mask": attention_mask[:, :self.max_length],
        }

    def _measure_prefix_length(self,
                               conversation_slice: List[Dict[str, str]],
                               add_generation_prompt: bool = False) -> int:
        """
        Helper that tokenizes a slice of the conversation to compute offsets.
        """
        if len(conversation_slice) == 0:
            return 0
        if hasattr(self.tokenizer, "apply_chat_template"):
            tokens = self.tokenizer.apply_chat_template(
                conversation=conversation_slice,
                tokenize=True,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.max_length,
                add_generation_prompt=add_generation_prompt,
            )
            return tokens.shape[1]
        else:
            joined = self._convert_messages_to_text(
                conversation_slice,
                add_generation_prompt=add_generation_prompt)
            tokens = self.tokenizer(
                joined,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )["input_ids"]
            return tokens.shape[1]

    def _convert_messages_to_text(self,
                                  messages: List[Dict[str, str]],
                                  add_generation_prompt: bool = False) -> str:
        """
        Fallback conversion for tokenizers without ``apply_chat_template``.

        Mirrors Open-Instruct's ``convert_messages_to_text`` utility:
        represent each turn as ``User: ...`` / ``Assistant: ...`` (see
        scripts/data/filtering_and_updates/filter_special_tokens.py in
        allenai/open-instruct).
        """
        text_parts = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role", "").lower()
            content = message.get("content", "")
            if not content:
                continue
            if role in ["system"]:
                text_parts.append(f"System: {content}")
            elif role in ["user", "human"]:
                text_parts.append(f"User: {content}")
            elif role in ["assistant", "ai"]:
                text_parts.append(f"Assistant: {content}")
            else:
                text_parts.append(f"{role.capitalize()}: {content}")

        if add_generation_prompt:
            text_parts.append("Assistant:")
        return "\n\n".join(text_parts)
