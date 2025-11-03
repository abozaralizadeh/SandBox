import logging
import math
from typing import List, Optional, Sequence

try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    tiktoken = None

from pydantic import PrivateAttr

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import AzureChatOpenAI


logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def _flatten_message_content(content: object) -> str:
    """Convert tool/content payloads into a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return "\n".join(_flatten_message_content(item) for item in content)
    if isinstance(content, dict):
        return "\n".join(
            f"{key}: {_flatten_message_content(value)}" for key, value in content.items()
        )
    return str(content)


class TokenAwareAzureChatOpenAI(AzureChatOpenAI):
    """AzureChatOpenAI wrapper that keeps inputs within model limits.

    The wrapper performs three stages before each model invocation:
      1. Summarize oversized tool outputs using a map-reduce strategy.
      2. Summarize old conversational history while keeping the most recent turns.
      3. Ensure the final prompt stays below ``max_input_tokens``.
    """

    _max_input_tokens: int = PrivateAttr(default=240_000)
    _tool_message_token_limit: int = PrivateAttr(default=80_000)
    _summary_chunk_tokens: int = PrivateAttr(default=16_000)
    _keep_last_messages: int = PrivateAttr(default=6)
    _max_summary_rounds: int = PrivateAttr(default=3)
    _encoding_cache = PrivateAttr(default=None)

    def __init__(
        self,
        *args,
        max_input_tokens: int = 240_000,
        tool_message_token_limit: int = 80_000,
        summary_chunk_tokens: int = 16_000,
        keep_last_messages: int = 6,
        max_summary_rounds: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._max_input_tokens = max_input_tokens
        self._tool_message_token_limit = tool_message_token_limit
        self._summary_chunk_tokens = summary_chunk_tokens
        self._keep_last_messages = keep_last_messages
        self._max_summary_rounds = max_summary_rounds
        self._encoding_cache = None

    async def _agenerate(  # type: ignore[override]
        self,
        messages: Sequence[BaseMessage],
        *,
        stop: Optional[Sequence[str]] = None,
        run_manager=None,
        **kwargs,
    ):
        processed_messages = await self._prepare_messages(messages, run_manager)
        approx_input_tokens = self._count_messages_tokens(processed_messages)
        logger.info(
            "LLM call starting | messages=%d | approx_input_tokens=%d",
            len(processed_messages),
            approx_input_tokens,
        )
        result = await super()._agenerate(
            processed_messages, stop=stop, run_manager=run_manager, **kwargs
        )
        token_usage = {}
        if hasattr(result, "llm_output") and result.llm_output:
            token_usage = result.llm_output.get("token_usage") or result.llm_output.get(
                "usage", {}
            )
        first_generation = result.generations[0]
        if hasattr(first_generation, "message"):
            output_text = first_generation.message.content or ""
        else:
            output_text = first_generation[0].message.content  # type: ignore[index]
        completion_tokens = (
            token_usage.get("completion_tokens")
            or token_usage.get("total_tokens")
            or 0
        )
        logger.info(
            "LLM call finished | approx_input_tokens=%d | completion_tokens=%s | output_chars=%d",
            approx_input_tokens,
            completion_tokens,
            len(output_text),
        )
        return result

    async def _prepare_messages(
        self, messages: Sequence[BaseMessage], run_manager
    ) -> Sequence[BaseMessage]:
        messages = list(messages)
        logger.info(
            "Preparing messages | original_count=%d | approx_tokens=%d",
            len(messages),
            self._count_messages_tokens(messages),
        )
        messages = await self._compress_tool_messages(messages, run_manager)
        total_tokens = self._count_messages_tokens(messages)
        if total_tokens <= self._max_input_tokens:
            logger.info(
                "Preparation finished without history summarisation | message_count=%d | approx_tokens=%d",
                len(messages),
                total_tokens,
            )
            return messages

        messages = await self._summarise_history(messages, run_manager)
        total_tokens = self._count_messages_tokens(messages)
        if total_tokens <= self._max_input_tokens:
            logger.info(
                "Preparation finished after history summarisation | message_count=%d | approx_tokens=%d",
                len(messages),
                total_tokens,
            )
            return messages

        # As a final fallback, keep trimming the earliest non-system messages.
        return self._trim_messages_to_limit(messages)

    async def _compress_tool_messages(
        self, messages: Sequence[BaseMessage], run_manager
    ) -> Sequence[BaseMessage]:
        compressed: List[BaseMessage] = []
        for message in messages:
            if isinstance(message, ToolMessage):
                flattened = _flatten_message_content(message.content)
                if self._approx_tokens(flattened) > self._tool_message_token_limit:
                    logger.info(
                        "Summarising large tool output | approx_tokens=%d | limit=%d",
                        self._approx_tokens(flattened),
                        self._tool_message_token_limit,
                    )
                    summary = await self._summarise_text(
                        flattened, run_manager=run_manager
                    )
                    logger.info(
                        "Tool output compressed | new_tokens=%d",
                        self._approx_tokens(summary),
                    )
                    compressed.append(
                        ToolMessage(
                            content=summary,
                            tool_call_id=message.tool_call_id or "",
                            additional_kwargs=getattr(
                                message, "additional_kwargs", {}
                            ),
                            name=getattr(message, "name", None),
                        )
                    )
                    continue
            compressed.append(message)
        return compressed

    async def _summarise_history(
        self, messages: Sequence[BaseMessage], run_manager
    ) -> Sequence[BaseMessage]:
        if len(messages) <= self._keep_last_messages + 2:
            return messages

        system_messages = [m for m in messages if isinstance(m, SystemMessage)]
        non_system = [m for m in messages if not isinstance(m, SystemMessage)]
        if len(non_system) <= self._keep_last_messages:
            return messages

        keep_last = self._keep_last_messages
        while keep_last > 0:
            head = non_system[:-keep_last]
            tail = non_system[-keep_last:]
            if not head:
                break
            summary_source = self._messages_to_text(head)
            summary = await self._summarise_text(
                summary_source, run_manager=run_manager
            )
            summary_message = SystemMessage(
                content="Summary of earlier conversation:\n" + summary
            )
            candidate = system_messages + [summary_message] + tail
            if self._count_messages_tokens(candidate) <= self._max_input_tokens:
                logger.info(
                    "History summarised | kept_messages=%d | summary_tokens=%d",
                    len(tail),
                    self._approx_tokens(summary),
                )
                return candidate
            keep_last -= 1

        # If we reach here, summarising still exceeds the limit and we fall back
        # to trimming in the caller.
        fallback_keep = max(1, keep_last)
        logger.warning(
            "Falling back to trimming messages | keeping_last=%d", fallback_keep
        )
        return system_messages + non_system[-fallback_keep:]

    def _trim_messages_to_limit(
        self, messages: Sequence[BaseMessage]
    ) -> Sequence[BaseMessage]:
        trimmed = list(messages)
        idx = 0
        while trimmed and self._count_messages_tokens(trimmed) > self._max_input_tokens:
            # Skip system messages to preserve instructions.
            if isinstance(trimmed[idx], SystemMessage):
                idx += 1
                if idx >= len(trimmed):
                    break
                continue
            trimmed.pop(idx)
            idx = min(idx, len(trimmed) - 1)
        logger.warning(
            "Messages trimmed to fit token budget | final_count=%d | approx_tokens=%d",
            len(trimmed),
            self._count_messages_tokens(trimmed),
        )
        return trimmed

    async def _summarise_text(
        self,
        text: str,
        *,
        run_manager,
        chunk_token_limit: Optional[int] = None,
        round_number: int = 0,
    ) -> str:
        chunk_limit = chunk_token_limit or self._summary_chunk_tokens
        if self._approx_tokens(text) <= chunk_limit:
            return text.strip()

        chunks = self._chunk_text(text, chunk_limit)
        partial_summaries: List[str] = []
        total_chunks = len(chunks)
        logger.info(
            "Summarising text | total_chunks=%d | chunk_limit=%d",
            total_chunks,
            chunk_limit,
        )
        for index, chunk in enumerate(chunks, start=1):
            prompt: List[BaseMessage] = [
                SystemMessage(
                    content=(
                        "You are a focused scientific summariser. Compress the text "
                        "into dense, factual bullet points. Preserve critical numbers, "
                        "methods, and citations when present. "
                        "Output no more than 250 words."
                    )
                ),
                HumanMessage(
                    content=f"Chunk {index}/{total_chunks}:\n{chunk.strip()}"
                ),
            ]
            logger.info(
                "Summarising chunk %d/%d | approx_tokens=%d",
                index,
                total_chunks,
                self._approx_tokens(chunk),
            )
            result = await super()._agenerate(
                prompt, run_manager=run_manager, stop=None
            )
            first_generation = result.generations[0]
            if hasattr(first_generation, "message"):
                summary_content = first_generation.message.content
            else:
                summary_content = first_generation[0].message.content  # type: ignore[index]
            summary = summary_content.strip()
            partial_summaries.append(summary)
            logger.info(
                "Chunk %d/%d summary tokens=%d",
                index,
                total_chunks,
                self._approx_tokens(summary),
            )

        combined = "\n\n".join(partial_summaries)
        if self._approx_tokens(combined) <= chunk_limit:
            return combined
        # Reduce step: summarise combined partial summaries recursively.
        if round_number >= self._max_summary_rounds:
            approx_chars = max(1, chunk_limit * 4)
            logger.warning(
                "Reached max summary rounds; truncating output | approx_chars=%d",
                approx_chars,
            )
            return combined[:approx_chars].strip()
        return await self._summarise_text(
            combined,
            run_manager=run_manager,
            chunk_token_limit=chunk_limit,
            round_number=round_number + 1,
        )

    def _chunk_text(self, text: str, chunk_token_limit: int) -> List[str]:
        if not text:
            return [text]
        approx_chars = max(1, chunk_token_limit * 4)
        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = start + approx_chars
            # Attempt to split on a nearby paragraph boundary.
            if end < len(text):
                newline_pos = text.rfind("\n\n", start, end)
                if newline_pos != -1 and newline_pos > start + approx_chars // 2:
                    end = newline_pos
            chunks.append(text[start:end])
            start = end
        return chunks

    def _messages_to_text(self, messages: Sequence[BaseMessage]) -> str:
        lines: List[str] = []
        for message in messages:
            role = message.type.upper()
            content = _flatten_message_content(message.content)
            lines.append(f"{role}:\n{content.strip()}")
        return "\n\n".join(lines)

    def _count_messages_tokens(self, messages: Sequence[BaseMessage]) -> int:
        return sum(self._count_single_message_tokens(message) for message in messages)

    def _count_single_message_tokens(self, message: BaseMessage) -> int:
        base = 6  # rough overhead for metadata
        content = _flatten_message_content(message.content)
        return base + self._approx_tokens(content)

    def _approx_tokens(self, text: str) -> int:
        if not text:
            return 0
        if tiktoken:
            encoding = self._get_encoding()
            return len(encoding.encode(text))
        # Fallback heuristic assuming ~4 characters per token.
        return max(1, math.ceil(len(text) / 4))

    def _get_encoding(self):
        if self._encoding_cache is not None:
            return self._encoding_cache
        model = getattr(self, "model_name", None) or getattr(
            self, "azure_deployment", "gpt-4o"
        )
        try:
            self._encoding_cache = tiktoken.encoding_for_model(model)
        except Exception:
            self._encoding_cache = tiktoken.get_encoding("cl100k_base")
        return self._encoding_cache
