"""
Complete example: Vocabulary Tutor agent using LangChain + WordOrb.

This agent can:
  1. Look up any word and explain its meaning, pronunciation, and etymology.
  2. Retrieve structured lessons for any day/track/age combination.
  3. Generate quiz questions and evaluate student answers.
  4. Explore the knowledge graph for thematic word connections.

Run::

    export OPENAI_API_KEY="sk-..."
    export WORDORB_API_KEY="wo_..."
    python example_agent.py

Requires: langchain, langchain-openai, httpx
"""

from __future__ import annotations

import os
import sys

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from wordorb_tools import WordOrbToolkit

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WORDORB_API_KEY = os.environ.get("WORDORB_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

if not WORDORB_API_KEY:
    print("Error: Set the WORDORB_API_KEY environment variable.")
    print("  Get a free key at https://wordorb.ai/pricing")
    sys.exit(1)

if not OPENAI_API_KEY:
    print("Error: Set the OPENAI_API_KEY environment variable.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Build the agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert vocabulary tutor powered by the WordOrb platform.

Your capabilities:
- Look up any English word for its definition, pronunciation, etymology, and translations.
- Retrieve structured 5-phase lessons (hook, story, wonder, action, wisdom).
- Generate and administer quiz questions (multiple choice, true/false, open-ended, etc.).
- Explore the knowledge graph to find thematic connections between words and lessons.

Teaching style:
- Start by understanding the student's level and goals.
- When teaching a word, always include pronunciation (IPA) and at least one example.
- Connect new words to ones the student already knows using the knowledge graph.
- After teaching, offer a short quiz to reinforce learning.
- Be encouraging, clear, and adapt explanations to the student's age group.

Always use the WordOrb tools for accurate, curriculum-aligned data.  Never invent
definitions or quiz questions -- rely on the API.
"""


def build_agent() -> AgentExecutor:
    """Construct the vocabulary tutor agent with WordOrb tools."""

    # 1. Tools
    toolkit = WordOrbToolkit(api_key=WORDORB_API_KEY)
    tools = toolkit.get_tools()

    # 2. LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

    # 3. Prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # 4. Agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=10,
        handle_parsing_errors=True,
    )


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------


def main() -> None:
    """Run an interactive vocabulary tutor session."""
    agent = build_agent()
    chat_history: list = []

    print("=" * 60)
    print("  WordOrb Vocabulary Tutor")
    print("  Type 'quit' to exit.")
    print("=" * 60)
    print()

    # Seed questions the user might try
    print("Try asking:")
    print('  - "What does serendipity mean?"')
    print('  - "Give me today\'s lesson on the learn track"')
    print('  - "Quiz me on 5 multiple choice questions"')
    print('  - "What words are related to courage?"')
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        result = agent.invoke(
            {"input": user_input, "chat_history": chat_history}
        )

        output = result.get("output", "")
        print(f"\nTutor: {output}\n")

        # Maintain chat history for multi-turn context
        chat_history.append(HumanMessage(content=user_input))
        from langchain_core.messages import AIMessage

        chat_history.append(AIMessage(content=output))


if __name__ == "__main__":
    main()
