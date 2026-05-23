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

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# =========================================================
# RESULT DATA STRUCTURE
# =========================================================

@dataclass
class LLMResult:

    answer: str

    model_used: str

    generation_time_ms: float

    fallback_used: bool

    prompt_tokens_est: int


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are SWARA, a grounded AI research assistant specialized in document understanding, contextual reasoning, narrative interpretation, and evidence-based synthesis.

You answer questions using ONLY the uploaded document content and retrieved evidence.

PRIMARY OBJECTIVE:
Generate clear, natural, insightful, and well-grounded answers by synthesizing information across relevant document excerpts.

GROUNDING RULES:
1. Never fabricate facts not supported by the uploaded documents.
2. Never use outside knowledge.
3. If evidence is insufficient, clearly acknowledge uncertainty.
4. Avoid unsupported assumptions or speculation.
5. Keep all interpretations grounded in textual evidence.
6. Prefer synthesis over direct extraction.

CRITICAL RESPONSE RULES:
1. NEVER mention:
   - retrieved context
   - retrieved chunks
   - evidence numbers
   - source numbering
   - internal retrieval systems
   - semantic search
2. NEVER say phrases like:
   - "Based on the retrieved context..."
   - "According to the context..."
   - "Evidence 1 suggests..."
   - "The provided information states..."
3. Speak naturally and directly as if you genuinely understood the document.
4. Do not expose internal reasoning structure.

SYNTHESIS RULES:
1. Combine related details naturally across multiple excerpts.
2. Infer motivations, emotions, relationships, and themes ONLY when strongly supported.
3. Preserve ambiguity when evidence is incomplete.
4. Avoid robotic summarization.
5. Prioritize coherent interpretation over fragmented observations.
6. Sound analytical but natural.

CONVERSATIONAL CONTINUITY:
1. Use recent conversation history carefully.
2. Resolve references like:
   - he
   - she
   - they
   - him
   - her
   - them
   - this event
   - that decision
   - that scene
3. Maintain continuity across follow-up questions.
4. Assume follow-up questions refer to previously discussed entities unless context suggests otherwise.

ANSWER STYLE:
1. Use concise but complete explanations.
2. Use short readable paragraphs.
3. Use bullet points only when useful.
4. Avoid repetitive wording.
5. Avoid excessive hedging.
6. Avoid over-explaining obvious details.
7. Sound thoughtful, grounded, and conversational.
8. Focus on interpretation and synthesis rather than quoting.

FAILURE HANDLING:
If the uploaded documents do not contain enough information to answer the question, respond EXACTLY with:

"The uploaded documents do not contain sufficient relevant information to answer this question."

Retrieved Context:
{context}
"""

# =========================================================
# LLM SERVICE
# =========================================================

class LLMService:

    def __init__(
        self,
        api_key: str | None = None,
        primary_model: str | None = None,
        fallback_model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 1500,
    ):

        self._api_key = (
            api_key
            or settings.groq_api_key
        )

        self._primary_model = (
            primary_model
            or settings.groq_primary_model
        )

        self._fallback_model = (
            fallback_model
            or settings.groq_fallback_model
        )

        self._temperature = temperature

        self._max_tokens = max_tokens

        if not self._api_key:

            logger.warning(
                "GROQ_API_KEY is not configured."
            )

        self._primary_client = (
            self._build_client(
                self._primary_model
            )
        )

        self._fallback_client = (
            self._build_client(
                self._fallback_model
            )
        )

        logger.info(
            f"LLMService initialized | "
            f"primary={self._primary_model} | "
            f"fallback={self._fallback_model}"
        )

    # =====================================================
    # PUBLIC GENERATION API
    # =====================================================

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

        # =================================================
        # PRIMARY MODEL
        # =================================================

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

            return LLMResult(
                answer=answer.strip(),
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
                f"Primary model failed: "
                f"{str(primary_error)}"
            )

        # =================================================
        # FALLBACK MODEL
        # =================================================

        try:

            answer = self._invoke_with_retry(
                self._fallback_client,
                messages,
            )

            elapsed_ms = (
                time.perf_counter() - start
            ) * 1000

            return LLMResult(
                answer=answer.strip(),
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
                    "⚠️ The AI answer service is "
                    "temporarily unavailable."
                ),
                model_used="none",
                generation_time_ms=elapsed_ms,
                fallback_used=True,
                prompt_tokens_est=0,
            )

    # =====================================================
    # CLIENT CREATION
    # =====================================================

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

    # =====================================================
    # MESSAGE BUILDING
    # =====================================================

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

                # =========================================
                # SUPPORT BOTH DICTS + PYDANTIC OBJECTS
                # =========================================

                if isinstance(msg, dict):

                    role = msg.get(
                        "role",
                        "user",
                    )

                    content = msg.get(
                        "content",
                        "",
                    )

                else:

                    role = getattr(
                        msg,
                        "role",
                        "user",
                    )

                    content = getattr(
                        msg,
                        "content",
                        "",
                    )

                formatted_turns.append(
                    f"{role.upper()}: {content}"
                )

            conversation_text = (
                "RECENT CONVERSATION:\n\n"
                + "\n".join(formatted_turns)
            )

        system_content = SYSTEM_PROMPT.format(
            context=context
        )

        human_content = f"""
{conversation_text}

CURRENT QUESTION:
{question}

IMPORTANT:
Resolve ambiguous references using recent conversation history when appropriate.
"""

        return [
            SystemMessage(
                content=system_content
            ),
            HumanMessage(
                content=human_content
            ),
        ]

    # =====================================================
    # RETRY LOGIC
    # =====================================================

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

    # =====================================================
    # TOKEN ESTIMATION
    # =====================================================

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


# =========================================================
# SINGLETON FACTORY
# =========================================================

@lru_cache(maxsize=1)
def get_llm_service() -> LLMService:

    return LLMService()