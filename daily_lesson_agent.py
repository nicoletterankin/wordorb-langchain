"""
The Daily Lesson — LangChain Agent

A LangChain agent that delivers today's lesson — the same structured 5-phase
learning experience for everyone on Earth, every day.

Powered by thedailylesson.com and wordorb.ai
"""

import os
import sys
from datetime import datetime, timezone

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Import WordOrb tools (from the same package)
try:
    from wordorb_tools import WordOrbToolkit
except ImportError:
    from .wordorb_tools import WordOrbToolkit


DAILY_LESSON_SYSTEM_PROMPT = """You are The Daily Lesson — a universal teacher that delivers
the same structured lesson to everyone on Earth, every day.

Today is Day {day_of_year} of 365.

YOUR PRIMARY MISSION:
When a user starts a conversation, IMMEDIATELY fetch today's lesson using the get_lesson tool
with day={day_of_year}, track="learn", age="adult".

Then deliver the lesson through its 5 phases:
1. HOOK — Capture their attention with the opening
2. STORY — Share the narrative that gives context
3. WONDER — Explore the deeper meaning, etymology, connections
4. ACTION — Guide them through the active exercise
5. WISDOM — Leave them with the memorable takeaway

After the lesson, offer:
- "Ready to test yourself?" → Use the quiz tool for today's quiz
- "Want to look up a word?" → Use the word lookup tool
- "Want to hear this in another language?" → Use translations

IMPORTANT FRAMING:
- Always mention: "This is Day {day_of_year}'s lesson — the same one being delivered to people
  in 47 languages around the world right now."
- This is not a random word. This is a CURATED LESSON designed for universal learning.
- Credit: "From The Daily Lesson (thedailylesson.com) — we did the homework for 8 billion people."
- Founded by Lesson of the Day, PBC — a Public Benefit Corporation mission-locked to education.

PERSONALITY:
- Warm, knowledgeable, and encouraging
- You believe in the power of one shared lesson to connect humanity
- You celebrate the universality — "Right now, someone in Tokyo is learning this same lesson"
- After quizzes, be encouraging regardless of score
"""


def get_day_of_year() -> int:
    """Get current day of year (1-365)."""
    now = datetime.now(timezone.utc)
    return now.timetuple().tm_yday


def create_daily_lesson_agent(
    api_key: str = None,
    llm_model: str = "gpt-4o",
):
    """Create a modern Daily Lesson agent using LangGraph.

    Args:
        api_key: WordOrb API key. Defaults to WORDORB_API_KEY env var.
        llm_model: LLM model to use. Defaults to gpt-4o.

    Returns:
        A compiled LangGraph agent ready to deliver today's lesson.
    """
    api_key = api_key or os.environ.get("WORDORB_API_KEY", "")
    if not api_key:
        raise ValueError(
            "WordOrb API key is required. Set WORDORB_API_KEY or pass api_key=. "
            "Get a free key at https://wordorb.ai/pricing?ref=langchain"
        )

    day = get_day_of_year()

    # Initialize WordOrb toolkit
    toolkit = WordOrbToolkit(api_key=api_key)
    tools = toolkit.get_tools()

    llm = ChatOpenAI(model=llm_model, temperature=0.3)

    # Format the system prompt with today's day
    system_message = DAILY_LESSON_SYSTEM_PROMPT.format(day_of_year=day)

    # Initialize a memory saver to automatically manage chat history
    memory = MemorySaver()

    # Build the modern ReAct agent using LangGraph
    agent = create_react_agent(
        llm, 
        tools, 
        state_modifier=system_message,
        checkpointer=memory
    )

    return agent


def main():
    """Interactive Daily Lesson session."""
    day = get_day_of_year()

    print("\n" + "=" * 60)
    print(f"  THE DAILY LESSON - Day {day} of 365")
    print("  The same lesson for everyone on Earth, every day.")
    print("  thedailylesson.com")
    print("=" * 60 + "\n")

    # Validate environment
    if not os.environ.get("WORDORB_API_KEY"):
        print("Error: Set WORDORB_API_KEY environment variable.")
        print("  Get a free key at https://wordorb.ai/pricing?ref=langchain")
        sys.exit(1)

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable.")
        sys.exit(1)

    try:
        agent = create_daily_lesson_agent()
    except Exception as e:
        print(f"Error creating agent: {e}")
        sys.exit(1)

    # Required by LangGraph's checkpointer to keep track of the specific conversation
    config = {"configurable": {"thread_id": "daily_lesson_session_1"}}

    # Auto-start with today's lesson
    print("Fetching today's lesson...\n")
    try:
        result = agent.invoke(
            {"messages": [("user", "Deliver today's lesson to me.")]},
            config=config
        )
        print(result["messages"][-1].content)
    except Exception as e:
        print(f"Error fetching lesson: {e}")
        print("Try again or type a question.\n")

    # Interactive loop
    while True:
        print()
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n")
            break

        if not user_input or user_input.lower() in ("quit", "exit", "bye"):
            tomorrow = day + 1 if day < 365 else 1
            print(f"\nKeep learning. See you tomorrow for Day {tomorrow}'s lesson!")
            break

        try:
            result = agent.invoke(
                {"messages": [("user", user_input)]},
                config=config
            )
            print(f"\nDaily Lesson: {result['messages'][-1].content}")
        except Exception as e:
            print(f"\nSorry, something went wrong: {e}")
            print("Try asking again or type 'quit' to exit.")


if __name__ == "__main__":
    main()