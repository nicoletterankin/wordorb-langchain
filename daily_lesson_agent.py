"""
The Daily Lesson — LangChain Agent

A LangChain agent that delivers today's lesson — the same structured 5-phase
learning experience for everyone on Earth, every day.

Not a word of the day. A LESSON of the day.
365 lessons. 47 languages. Every device. Every age.

Powered by thedailylesson.com and wordorb.ai
"""

import os
import sys
from datetime import datetime, timezone
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

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
    verbose: bool = False,
    max_iterations: int = 10,
) -> AgentExecutor:
    """Create a Daily Lesson agent with WordOrb tools.

    Args:
        api_key: WordOrb API key. Defaults to WORDORB_API_KEY env var.
        llm_model: LLM model to use. Defaults to gpt-4o.
        verbose: Whether to show agent reasoning.

    Returns:
        AgentExecutor ready to deliver today's lesson.
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

    # Build the prompt with today's day
    prompt = ChatPromptTemplate.from_messages([
        ("system", DAILY_LESSON_SYSTEM_PROMPT.format(day_of_year=day)),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    llm = ChatOpenAI(model=llm_model, temperature=0.3)
    agent = create_openai_tools_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        max_iterations=max_iterations,
        handle_parsing_errors=True,
        return_intermediate_steps=False,
    )


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
        agent = create_daily_lesson_agent(verbose=False)
    except Exception as e:
        print(f"Error creating agent: {e}")
        sys.exit(1)

    chat_history = []

    # Auto-start with today's lesson
    print("Fetching today's lesson...\n")
    try:
        result = agent.invoke({
            "input": "Deliver today's lesson to me.",
            "chat_history": chat_history,
        })
        output = result["output"]
        print(output)
        chat_history.append(HumanMessage(content="Deliver today's lesson to me."))
        chat_history.append(AIMessage(content=output))
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
            result = agent.invoke({
                "input": user_input,
                "chat_history": chat_history,
            })
            output = result["output"]
            print(f"\nDaily Lesson: {output}")
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=output))
        except Exception as e:
            print(f"\nSorry, something went wrong: {e}")
            print("Try asking again or type 'quit' to exit.")


if __name__ == "__main__":
    main()
