"""
app/services/llm_service.py
============================
LLM answer generation service using Groq API + LangChain.
"""

import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────
# Result Data Structure
# ─────────────────────────────────────────────

@dataclass
class LLMResult:
    answer: str
    model_used: str
    generation_time_ms: float
    fallback_used: bool
    prompt_tokens_est: int


# ─────────────────────────────────────────────
# FINAL CALIBRATED SYSTEM PROMPT
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """
You are SWARA, a grounded AI research assistant specialized in document understanding, synthesis, and contextual reasoning.

You must answer questions using ONLY the retrieved document context.

PRIMARY OBJECTIVE:
Provide accurate, coherent, evidence-grounded answers by synthesizing information across retrieved context chunks.

GROUNDING RULES:
1. Never fabricate facts not supported by the retrieved context.
2. Never use outside knowledge.
3. If evidence is insufficient, explicitly acknowledge the limitation.
4. Avoid speculation or unsupported assumptions.
5. Maintain factual consistency with retrieved evidence.
6. Prefer grounded synthesis over isolated sentence extraction.

SYNTHESIS RULES:
1. Combine related evidence across multiple retrieved chunks.
2. Produce coherent summaries instead of disconnected observations.
3. Connect narrative, emotional, or thematic details when clearly supported by the retrieved context.
4. Identify motivations, emotional states, relationships, and thematic patterns when multiple pieces of evidence consistently support them.
5. Use careful interpretation when strongly supported by retrieved evidence.
6. Avoid overly defensive phrasing when reasonable evidence-based synthesis is possible.

CONVERSATIONAL CONTINUITY:
1. Use recent conversation history to maintain continuity across follow-up questions.
2. Resolve references and pronouns using the ongoing discussion context.
3. Avoid contradicting earlier grounded answers unless new retrieved evidence changes the interpretation.
4. Preserve conversational coherence across turns.

ANSWER STYLE:
- Use concise but complete explanations.
- Use short paragraphs.
- Use bullet points where useful.
- Avoid repetitive wording.
- Focus on clarity, synthesis, and evidence-grounded reasoning.
- Sound analytical and confident when evidence is strong.
- Clearly acknowledge uncertainty when evidence is weak.

FAILURE HANDLING:
If the retrieved context does not sufficiently support the answer, respond clearly with:

"The uploaded documents do not contain sufficient relevant information to answer this question."

The following context was retrieved from uploaded documents:

{context}
"""

# ─────────────────────────────────────────────
# HUMAN TEMPLATE
# ─────────────────────────────────────────────

HUMAN_TEMPLATE = "Question: {question}"


# ─────────────────────────────────────────────
# LLM SERVICE
# ─────────────────────────────────────────────

class LLMService:

    def __init__(
        self,
        api_key: str | None = None,
        primary_model: str | None = None,
        fallback_model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ):

        self._api_key = api_key or settings.groq_api_key
        self._primary_model = (
            primary_model or settings.groq_primary_model
        )
        self._fallback_model = (
            fallback_model or settings.groq_fallback_model
        )

        self._temperature = temperature
        self._max_tokens = max_tokens

        if not self._api_key:

            logger.warning(
                "GROQ_API_KEY is not configured."
            )

        self._primary_client = self._build_client(
            self._primary_model
        )

        self._fallback_client = self._build_client(
            self._fallback_model
        )

        logger.info(
            f"LLMService initialized | "
            f"primary={self._primary_model} | "
            f"fallback={self._fallback_model} | "
            f"temperature={self._temperature}"
        )

    # ─────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────

    def generate(
        self,
        question: str,
        context: str,
        chat_history: Optional[list] = None,
    ) -> LLMResult:

        if not self._api_key:

            return LLMResult(
                answer=(
                    "⚠️ Groq API key is not configured. "
                    "Please add GROQ_API_KEY to your .env file."
                ),
                model_used="none",
                generation_time_ms=0.0,
                fallback_used=False,
                prompt_tokens_est=0,
            )

        messages = self._build_messages(
            question=question,
            context=context,
            chat_history=chat_history,
        )

        start = time.perf_counter()

        # PRIMARY MODEL
        try:

            logger.info(
                f"LLM generation starting | "
                f"model={self._primary_model}",
                extra={"ai_pipeline": True},
            )

            answer = self._invoke_with_retry(
                self._primary_client,
                messages,
            )

            elapsed_ms = (
                time.perf_counter() - start
            ) * 1000

            logger.info(
                f"LLM generation complete | "
                f"model={self._primary_model} | "
                f"time={elapsed_ms:.0f}ms",
                extra={"ai_pipeline": True},
            )

            return LLMResult(
                answer=answer,
                model_used=self._primary_model,
                generation_time_ms=elapsed_ms,
                fallback_used=False,
                prompt_tokens_est=self._estimate_tokens(
                    question,
                    context,
                ),
            )

        except Exception as primary_error:

            logger.warning(
                f"Primary model failed | "
                f"error={primary_error}",
                extra={"ai_pipeline": True},
            )

        # FALLBACK MODEL
        try:

            answer = self._invoke_with_retry(
                self._fallback_client,
                messages,
            )

            elapsed_ms = (
                time.perf_counter() - start
            ) * 1000

            return LLMResult(
                answer=answer,
                model_used=self._fallback_model,
                generation_time_ms=elapsed_ms,
                fallback_used=True,
                prompt_tokens_est=self._estimate_tokens(
                    question,
                    context,
                ),
            )

        except Exception as fallback_error:

            elapsed_ms = (
                time.perf_counter() - start
            ) * 1000

            logger.error(
                f"Both models failed | "
                f"primary={primary_error} | "
                f"fallback={fallback_error}",
                extra={"ai_pipeline": True},
            )

            return LLMResult(
                answer=(
                    "⚠️ The AI answer service is temporarily unavailable."
                ),
                model_used="none",
                generation_time_ms=elapsed_ms,
                fallback_used=True,
                prompt_tokens_est=0,
            )

    # ─────────────────────────────────────────
    # CLIENT CREATION
    # ─────────────────────────────────────────

    def _build_client(
        self,
        model_name: str,
    ) -> ChatGroq:

        return ChatGroq(
            groq_api_key=self._api_key,
            model_name=model_name,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

    # ─────────────────────────────────────────
    # MESSAGE BUILDING
    # ─────────────────────────────────────────

    def _build_messages(
        self,
        question: str,
        context: str,
        chat_history: Optional[list] = None,
    ) -> list:

        conversation_text = ""

        if chat_history:

            recent_history = chat_history[-6:]

            formatted_turns = []

            for msg in recent_history:

                role = msg.get("role", "user")
                content = msg.get("content", "")

                formatted_turns.append(
                    f"{role.capitalize()}: {content}"
                )

            conversation_text = (
                "\n\nRecent Conversation:\n"
                + "\n".join(formatted_turns)
            )

        system_content = SYSTEM_PROMPT.format(
            context=context
        )

        human_content = (
            f"{conversation_text}\n\n"
            f"Current Question: {question}"
        )

        return [
            SystemMessage(content=system_content),
            HumanMessage(content=human_content),
        ]

    # ─────────────────────────────────────────
    # RETRY LOGIC
    # ─────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(2),
        wait=wait_exponential(
            multiplier=1,
            min=1,
            max=4,
        ),
        reraise=True,
    )
    def _invoke_with_retry(
        self,
        client: ChatGroq,
        messages: list,
    ) -> str:

        response = client.invoke(messages)

        return response.content

    # ─────────────────────────────────────────
    # TOKEN ESTIMATION
    # ─────────────────────────────────────────

    @staticmethod
    def _estimate_tokens(
        question: str,
        context: str,
    ) -> int:

        total_chars = (
            len(SYSTEM_PROMPT)
            + len(context)
            + len(question)
        )

        return total_chars // 4


# ─────────────────────────────────────────────
# SINGLETON FACTORY
# ─────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_llm_service() -> LLMService:

    return LLMService()