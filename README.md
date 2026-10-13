# NL2SQL — Natural Language to SQL Query Engine

A production-grade pipeline that converts plain English questions into SQL queries, executes them against a MySQL database, and returns readable results — with a Streamlit UI.

## Features

- **Multi-LLM support**: OpenAI GPT-4o, Anthropic Claude, Google Gemini, Ollama (local)
- **Smart prompt construction**: Auto-fetches DB schema, chunks large schemas, enriches context
- **MySQL execution**: Runs generated queries safely with read-only guard
- **Readable output**: Converts raw rows into a natural language summary + DataFrame
- **Streamlit UI**: Schema explorer, query history, result table + chart

---

## Project Structure

```
nl2sql/
├── app.py                  # Streamlit entry point
├── config/
│   └── settings.py         # All config (DB, LLM keys, params)
├── core/
│   ├── prompt_builder.py   # Schema chunking + prompt construction
│   ├── query_pipeline.py   # End-to-end NL → SQL → Result pipeline
│   └── result_formatter.py # Format raw DB results into readable text
├── db/
│   ├── connector.py        # MySQL connection + query executor
│   └── schema_fetcher.py   # Fetch and cache DB schema
├── llm/
│   ├── base.py             # Abstract LLM interface
│   ├── openai_llm.py       # OpenAI (GPT-4o) adapter
│   ├── anthropic_llm.py    # Anthropic (Claude) adapter
│   ├── gemini_llm.py       # Google Gemini adapter
│   ├── ollama_llm.py       # Ollama local models adapter
│   └── factory.py          # LLM factory — pick by name
├── utils/
│   └── logger.py           # Structured logging
├── requirements.txt
└── .env.example
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in your DB credentials and LLM API keys
```

### 3. Run the app

```bash
streamlit run app.py
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `DB_HOST` | MySQL host (default: localhost) |
| `DB_PORT` | MySQL port (default: 3306) |
| `DB_USER` | MySQL username |
| `DB_PASSWORD` | MySQL password |
| `DB_NAME` | Target database name |
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `GOOGLE_API_KEY` | Google Gemini API key |
| `OLLAMA_BASE_URL` | Ollama server URL (default: http://localhost:11434) |

---

## Supported LLMs

| Provider | Models |
|---|---|
| OpenAI | gpt-4o, gpt-4o-mini, gpt-3.5-turbo |
| Anthropic | claude-opus-4-6, claude-sonnet-4-6 |
| Google | gemini-3.5-flash, gemini-2.5-flash, gemini-2.5-pro |
| Ollama | llama3, mistral, codellama, any local model |

---

## Basic RAGAS / Text-to-SQL Evaluation

Create a CSV from `evaluation/dataset_template.csv` with:

- `question`: natural-language input
- `reference_sql`: verified SQL for the same database
- `case_id`: optional identifier

The evaluator reports:

- generated SQL execution success
- execution-result match against the reference SQL
- recall of reference tables in the retrieved schema context
- optional RAGAS SQL semantic equivalence

RAGAS is kept optional:

```bash
pip install -r requirements-eval.txt
```

Run the local execution and retrieval metrics:

```bash
python -m evaluation.run_evaluation evaluation/my_cases.csv \
  --provider openai --model gpt-4o-mini
```

Add `--ragas-semantic` to enable the optional LLM-judged SQL semantic score.
Database and API credentials are read from the existing environment settings.
