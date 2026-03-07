"""
LangChain tool definitions for the WordOrb.ai API.

Provides four tools that wrap the Orb Platform API (wordorb.ai):
  - WordOrbLookupTool  -- look up any English word
  - WordOrbLessonTool  -- get a structured 5-phase lesson
  - WordOrbQuizTool    -- get quiz / assessment questions
  - WordOrbGraphTool   -- query the word-to-lesson knowledge graph

Plus a convenience ``WordOrbToolkit`` that bundles all four.

Usage::

    from langchain_wordorb import WordOrbToolkit

    toolkit = WordOrbToolkit(api_key="wo_...")
    tools = toolkit.get_tools()

Requires ``langchain-core >= 0.2`` and ``httpx``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, ClassVar, Dict, List, Literal, Optional, Type, Union

import httpx
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "https://api.wordorb.ai"
DEFAULT_TIMEOUT = 30.0
VALID_TRACKS = ("learn", "grow", "teach", "trivia")
VALID_AGE_GROUPS = ("kid", "adult", "elder")
VALID_QUIZ_TYPES = ("quiz", "mc", "tf", "ab", "open", "reflection")

# ---------------------------------------------------------------------------
# Shared HTTP helpers
# ---------------------------------------------------------------------------


def _build_headers(api_key: str) -> Dict[str, str]:
    """Return authorization headers for the WordOrb API."""
    return {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "langchain-wordorb/1.0.0",
        "Accept": "application/json",
    }


def _handle_response(response: httpx.Response) -> Dict[str, Any]:
    """Parse a WordOrb API response, raising on HTTP errors."""
    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After", "60")
        raise RuntimeError(
            f"WordOrb rate limit exceeded. Retry after {retry_after}s. "
            "Consider upgrading your plan at https://wordorb.ai/pricing?ref=langchain"
        )
    if response.status_code == 401:
        raise RuntimeError(
            "WordOrb authentication failed. Check your API key. "
            "Get one at https://wordorb.ai/pricing?ref=langchain"
        )
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Pydantic argument schemas
# ---------------------------------------------------------------------------


class LookupWordInput(BaseModel):
    """Input schema for looking up a word."""

    word: str = Field(
        ...,
        description="The English word to look up (e.g. 'courage', 'serendipity').",
    )
    lang: Optional[str] = Field(
        default=None,
        description="ISO 639-1 language code for translations (e.g. 'es', 'ja'). "
        "Omit for English-only results.",
    )
    context: Optional[str] = Field(
        default=None,
        description="Optional sentence context for disambiguation.",
    )
    format: Optional[Literal["full", "minimal"]] = Field(
        default="full",
        description="'minimal' returns only word, definition, and IPA.",
    )


class GetLessonInput(BaseModel):
    """Input schema for retrieving a structured lesson."""

    day: int = Field(
        default=1,
        ge=1,
        le=365,
        description="Day of the year (1-365). Each day has unique lesson content.",
    )
    track: Literal["learn", "grow", "teach", "trivia"] = Field(
        default="learn",
        description="Learning track: learn (vocabulary), grow (personal development), "
        "teach (pedagogy), or trivia (fun facts).",
    )
    age: Literal["kid", "adult", "elder"] = Field(
        default="adult",
        description="Age group that controls content complexity.",
    )
    lang: str = Field(
        default="en",
        description="ISO 639-1 language code for the lesson content.",
    )


class GetQuizInput(BaseModel):
    """Input schema for retrieving quiz questions."""

    day: Optional[int] = Field(
        default=None,
        ge=1,
        le=365,
        description="Day of the year (1-365). Omit for random questions.",
    )
    track: Literal["learn", "grow", "teach", "trivia"] = Field(
        default="learn",
        description="Learning track.",
    )
    type: Optional[Literal["quiz", "mc", "tf", "ab", "open", "reflection"]] = Field(
        default=None,
        description="Question type filter: mc (multiple choice), tf (true/false), "
        "ab (A/B comparison), open (free response), reflection. Omit for all types.",
    )
    count: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of random questions to return (only used when day is omitted).",
    )


class GraphWordInput(BaseModel):
    """Input schema for querying the knowledge graph by word."""

    word: str = Field(
        ...,
        description="Word to look up in the knowledge graph.",
    )


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


class WordOrbLookupTool(BaseTool):
    """Look up any English word via the WordOrb dictionary API.

    Returns IPA pronunciation, definition, etymology, part of speech,
    example sentences, and translations in up to 47 languages.
    Backed by a 162K-word corpus.
    """

    name: str = "wordorb_lookup"
    description: str = (
        "Look up any English word. Returns IPA pronunciation, definition, "
        "etymology, part of speech, example sentences, and translations in "
        "47 languages. Use this when you need dictionary-quality word data."
    )
    args_schema: Type[BaseModel] = LookupWordInput

    api_key: str = Field(..., description="WordOrb API key (starts with wo_).")
    base_url: str = Field(default=DEFAULT_BASE_URL)
    timeout: float = Field(default=DEFAULT_TIMEOUT)

    def _run(
        self,
        word: str,
        lang: Optional[str] = None,
        context: Optional[str] = None,
        format: Optional[str] = "full",
    ) -> str:
        """Synchronous lookup."""
        params: Dict[str, str] = {}
        if lang:
            params["lang"] = lang
        if context:
            params["context"] = context
        if format == "minimal":
            params["format"] = "minimal"

        url = f"{self.base_url}/api/word/{word}"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(url, headers=_build_headers(self.api_key), params=params)
            data = _handle_response(resp)
        return _format_lookup(data)

    async def _arun(
        self,
        word: str,
        lang: Optional[str] = None,
        context: Optional[str] = None,
        format: Optional[str] = "full",
    ) -> str:
        """Asynchronous lookup."""
        params: Dict[str, str] = {}
        if lang:
            params["lang"] = lang
        if context:
            params["context"] = context
        if format == "minimal":
            params["format"] = "minimal"

        url = f"{self.base_url}/api/word/{word}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(
                url, headers=_build_headers(self.api_key), params=params
            )
            data = _handle_response(resp)
        return _format_lookup(data)


class WordOrbLessonTool(BaseTool):
    """Get a structured 5-phase vocabulary lesson from the Orb Platform.

    Each lesson has five phases: hook, story, wonder, action, wisdom.
    365 days x 4 tracks x 3 age groups = 4,380 unique lessons.
    """

    name: str = "wordorb_lesson"
    description: str = (
        "Get a structured vocabulary lesson with 5 phases (hook, story, wonder, "
        "action, wisdom). Specify day (1-365), track (learn/grow/teach/trivia), "
        "and age group (kid/adult/elder). Use this when building lesson plans "
        "or teaching vocabulary."
    )
    args_schema: Type[BaseModel] = GetLessonInput

    api_key: str = Field(..., description="WordOrb API key (starts with wo_).")
    base_url: str = Field(default=DEFAULT_BASE_URL)
    timeout: float = Field(default=DEFAULT_TIMEOUT)

    def _run(
        self,
        day: int = 1,
        track: str = "learn",
        age: str = "adult",
        lang: str = "en",
    ) -> str:
        """Synchronous lesson retrieval."""
        params = {"day": str(day), "track": track, "age": age, "lang": lang}
        url = f"{self.base_url}/api/lesson"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(url, headers=_build_headers(self.api_key), params=params)
            data = _handle_response(resp)
        return _format_lesson(data)

    async def _arun(
        self,
        day: int = 1,
        track: str = "learn",
        age: str = "adult",
        lang: str = "en",
    ) -> str:
        """Asynchronous lesson retrieval."""
        params = {"day": str(day), "track": track, "age": age, "lang": lang}
        url = f"{self.base_url}/api/lesson"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(
                url, headers=_build_headers(self.api_key), params=params
            )
            data = _handle_response(resp)
        return _format_lesson(data)


class WordOrbQuizTool(BaseTool):
    """Get quiz / assessment questions from the Orb Platform.

    Supports six question types: quiz, mc (multiple choice), tf (true/false),
    ab (A/B comparison), open (free response), and reflection.
    """

    name: str = "wordorb_quiz"
    description: str = (
        "Get vocabulary quiz questions. Optionally specify a day (1-365) for "
        "lesson-aligned questions, or omit for random questions. Filter by type: "
        "mc, tf, ab, open, reflection. Use this when assessing vocabulary knowledge."
    )
    args_schema: Type[BaseModel] = GetQuizInput

    api_key: str = Field(..., description="WordOrb API key (starts with wo_).")
    base_url: str = Field(default=DEFAULT_BASE_URL)
    timeout: float = Field(default=DEFAULT_TIMEOUT)

    def _run(
        self,
        day: Optional[int] = None,
        track: str = "learn",
        type: Optional[str] = None,
        count: int = 5,
    ) -> str:
        """Synchronous quiz retrieval."""
        if day is not None:
            params: Dict[str, str] = {"day": str(day), "track": track}
            if type:
                params["type"] = type
            url = f"{self.base_url}/api/quiz"
        else:
            params = {"track": track, "count": str(count)}
            if type:
                params["type"] = type
            url = f"{self.base_url}/api/quiz/random"

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(url, headers=_build_headers(self.api_key), params=params)
            data = _handle_response(resp)
        return _format_quiz(data)

    async def _arun(
        self,
        day: Optional[int] = None,
        track: str = "learn",
        type: Optional[str] = None,
        count: int = 5,
    ) -> str:
        """Asynchronous quiz retrieval."""
        if day is not None:
            params: Dict[str, str] = {"day": str(day), "track": track}
            if type:
                params["type"] = type
            url = f"{self.base_url}/api/quiz"
        else:
            params = {"track": track, "count": str(count)}
            if type:
                params["type"] = type
            url = f"{self.base_url}/api/quiz/random"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(
                url, headers=_build_headers(self.api_key), params=params
            )
            data = _handle_response(resp)
        return _format_quiz(data)


class WordOrbGraphTool(BaseTool):
    """Query the WordOrb knowledge graph to find word-to-lesson connections.

    Given a word, returns every lesson in which it appears, related words,
    and importance scores. Useful for building learning paths and finding
    thematic connections across the 365-day curriculum.
    """

    name: str = "wordorb_graph"
    description: str = (
        "Query the knowledge graph for a word. Returns all lessons where the "
        "word appears, related words, and importance scores. Use this to find "
        "thematic connections across the curriculum or build learning paths."
    )
    args_schema: Type[BaseModel] = GraphWordInput

    api_key: str = Field(..., description="WordOrb API key (starts with wo_).")
    base_url: str = Field(default=DEFAULT_BASE_URL)
    timeout: float = Field(default=DEFAULT_TIMEOUT)

    def _run(self, word: str) -> str:
        """Synchronous graph query."""
        params = {"word": word}
        url = f"{self.base_url}/api/graph/word"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(url, headers=_build_headers(self.api_key), params=params)
            data = _handle_response(resp)
        return _format_graph(data)

    async def _arun(self, word: str) -> str:
        """Asynchronous graph query."""
        params = {"word": word}
        url = f"{self.base_url}/api/graph/word"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(
                url, headers=_build_headers(self.api_key), params=params
            )
            data = _handle_response(resp)
        return _format_graph(data)


# ---------------------------------------------------------------------------
# Toolkit bundle
# ---------------------------------------------------------------------------


class WordOrbToolkit:
    """Convenience bundle that creates all four WordOrb tools.

    Usage::

        from langchain_wordorb import WordOrbToolkit

        toolkit = WordOrbToolkit(api_key="wo_live_abc123")
        tools = toolkit.get_tools()  # list of 4 BaseTool instances

    Parameters
    ----------
    api_key : str
        WordOrb API key (starts with ``wo_``).  Get one free at
        https://wordorb.ai/pricing?ref=langchain
    base_url : str, optional
        Override the API base URL.  Default: ``https://api.wordorb.ai``
    timeout : float, optional
        HTTP request timeout in seconds.  Default: 30.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        if not api_key:
            raise ValueError(
                "api_key is required. Get a free key at https://wordorb.ai/pricing?ref=langchain"
            )
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout

    def get_tools(self) -> List[BaseTool]:
        """Return a list of all WordOrb tools, ready to pass to an agent."""
        kwargs: Dict[str, Any] = {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "timeout": self.timeout,
        }
        return [
            WordOrbLookupTool(**kwargs),
            WordOrbLessonTool(**kwargs),
            WordOrbQuizTool(**kwargs),
            WordOrbGraphTool(**kwargs),
        ]


# ---------------------------------------------------------------------------
# Formatters  (tools return human-readable strings for the LLM)
# ---------------------------------------------------------------------------


def _format_lookup(data: Dict[str, Any]) -> str:
    """Format a word lookup response into a readable string."""
    parts = [f"Word: {data.get('word', 'unknown')}"]

    if data.get("ipa"):
        parts.append(f"IPA: {data['ipa']}")
    if data.get("pos"):
        parts.append(f"Part of speech: {data['pos']}")
    if data.get("definition") or data.get("def"):
        parts.append(f"Definition: {data.get('definition') or data.get('def')}")
    if data.get("etymology"):
        parts.append(f"Etymology: {data['etymology']}")
    if data.get("example"):
        parts.append(f"Example: {data['example']}")
    if data.get("synonyms"):
        syns = data["synonyms"]
        if isinstance(syns, list):
            syns = ", ".join(syns)
        parts.append(f"Synonyms: {syns}")
    if data.get("translations"):
        translations = data["translations"]
        if isinstance(translations, dict):
            tr_items = [f"  {k}: {v}" for k, v in list(translations.items())[:5]]
            parts.append("Translations:\n" + "\n".join(tr_items))

    return "\n".join(parts)


def _format_lesson(data: Dict[str, Any]) -> str:
    """Format a lesson response into a readable string."""
    parts = [
        f"Lesson: Day {data.get('day', '?')} | Track: {data.get('track', '?')} "
        f"| Age: {data.get('age_group', '?')}",
    ]
    if data.get("title"):
        parts.append(f"Title: {data['title']}")
    if data.get("theme"):
        parts.append(f"Theme: {data['theme']}")

    phases = data.get("phases", {})
    for phase_name in ("hook", "story", "wonder", "action", "wisdom"):
        content = phases.get(phase_name)
        if content:
            parts.append(f"\n--- {phase_name.upper()} ---\n{content}")

    return "\n".join(parts)


def _format_quiz(data: Dict[str, Any]) -> str:
    """Format a quiz response into a readable string."""
    interactions = data.get("interactions", [])
    if not interactions:
        return "No quiz questions found for the given parameters."

    parts = [
        f"Quiz: {data.get('count', len(interactions))} question(s) "
        f"| Track: {data.get('track', '?')}"
    ]

    for i, q in enumerate(interactions, 1):
        parts.append(f"\nQ{i} [{q.get('type', 'unknown')}]: {q.get('prompt', '')}")
        if q.get("choices"):
            choices = q["choices"]
            if isinstance(choices, list):
                for idx, c in enumerate(choices):
                    parts.append(f"  {chr(65 + idx)}. {c}")
        if q.get("answer"):
            parts.append(f"  Answer: {q['answer']}")
        if q.get("explanation"):
            parts.append(f"  Explanation: {q['explanation']}")

    return "\n".join(parts)


def _format_graph(data: Dict[str, Any]) -> str:
    """Format a graph query response into a readable string."""
    parts = [
        f"Knowledge Graph: '{data.get('word', '?')}' appears in "
        f"{data.get('appears_in', 0)} lesson(s)"
    ]

    lessons = data.get("lessons", [])
    for lesson in lessons[:10]:
        parts.append(
            f"  Day {lesson.get('day', '?')} ({lesson.get('track', '?')}) "
            f"- {lesson.get('phase', '?')} phase, "
            f"importance={lesson.get('importance', '?')}"
        )
        if lesson.get("context"):
            parts.append(f"    Context: {lesson['context']}")

    related = data.get("related_words", [])
    if related:
        parts.append(f"\nRelated words: {', '.join(related)}")

    return "\n".join(parts)
