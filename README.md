# langchain-wordorb

LangChain tools for the [WordOrb.ai](https://wordorb.ai) vocabulary and lesson API.

Provides four tools that give any LangChain agent access to 162K words,
226K structured lessons, 21K quiz interactions, and a 30K-edge knowledge graph.

## Installation

```bash
pip install langchain-wordorb
```

## Quick start

```python
from langchain_wordorb import WordOrbToolkit

toolkit = WordOrbToolkit(api_key="wo_live_abc123")
tools = toolkit.get_tools()

# Pass tools to any LangChain agent
from langchain.agents import AgentExecutor, create_openai_tools_agent
agent = create_openai_tools_agent(llm, tools, prompt)
```

## Tools

| Tool | Description |
|------|-------------|
| `wordorb_lookup` | Look up any English word -- IPA, definition, etymology, 47 languages |
| `wordorb_lesson` | 5-phase lesson (hook/story/wonder/action/wisdom), 365 days x 4 tracks x 3 ages |
| `wordorb_quiz` | Quiz questions: mc, tf, ab, open, reflection -- by day or random |
| `wordorb_graph` | Knowledge graph: word-to-lesson connections and related words |

## Individual tool usage

```python
from langchain_wordorb import WordOrbLookupTool

lookup = WordOrbLookupTool(api_key="wo_live_abc123")
result = lookup.invoke({"word": "serendipity"})
print(result)
```

## Async support

All tools support async via `ainvoke`:

```python
result = await lookup.ainvoke({"word": "courage"})
```

## Complete agent example

See `example_agent.py` for a full vocabulary tutor agent that uses all four
tools with GPT-4o.

```bash
export OPENAI_API_KEY="sk-..."
export WORDORB_API_KEY="wo_..."
python example_agent.py
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `api_key` | required | WordOrb API key (starts with `wo_`) |
| `base_url` | `https://api.wordorb.ai` | API base URL |
| `timeout` | `30.0` | HTTP timeout in seconds |

## Get an API key

Sign up at [wordorb.ai/pricing](https://wordorb.ai/pricing):

A key with no paid plan includes 10,000 calls per day, and no card is needed. See [wordorb.ai/pricing](https://wordorb.ai/pricing) for paid plans and current limits.

## Publishing to PyPI

```bash
# Build
pip install build twine
python -m build

# Test upload
twine upload --repository testpypi dist/*

# Production upload
twine upload dist/*
```

## LangChain Hub listing

To list on the [LangChain integrations directory](https://python.langchain.com/docs/integrations/):

1. Publish the package to PyPI as `langchain-wordorb`.
2. Open a PR to `langchain-ai/langchain` adding an entry under
   `docs/docs/integrations/tools/wordorb.ipynb`.
3. Include a working notebook that demonstrates all four tools.
4. Reference the PyPI package in the notebook metadata.

## License

MIT
