# 🤖 Multi-Agent Research Crew

An autonomous research system that takes any topic and produces a full executive report — completely free, no paid APIs required.

---

## What It Does

You type a research question. Six AI agents work in sequence to search the web, analyse findings, challenge assumptions, and write a polished report — saved as markdown files in the same folder as the script.

```
You → [Researcher] → [Consolidator] → [Analyst] → [Critic] → [Report Writer] → [Summary Writer]
                                                                      ↓                  ↓
                                                              final_report.md        summary.md
```

---

## Output Files

| File | Contents |
|---|---|
| `final_report.md` | Full executive report with findings, analysis, recommendations, and risks |
| `summary.md` | One-page summary: problem statement, top 3 findings, top 3 recommendations, biggest risk |

Both files are written to the **same folder as `research_crew.py`**.

---

## Requirements

- Python 3.10 or higher
- [Ollama](https://ollama.com) running locally (free, no API key)
- Internet connection (for DuckDuckGo search)

---

## Installation

### 1. Clone or download the project

```bash
git clone https://github.com/yourname/research-crew.git
cd research-crew
```

### 2. Install Python dependencies

```bash
pip install crewai ddgs requests beautifulsoup4 python-dotenv
```

> If `ddgs` doesn't install, try the older package name:
> ```bash
> pip install duckduckgo-search
> ```
> The script handles both automatically.

### 3. Install and start Ollama

Download from [ollama.com](https://ollama.com), then:

```bash
ollama serve                  # keep this terminal open
ollama pull llama3.2          # download the default model (~2 GB)
```

### 4. Create your `.env` file

Copy the example:

```bash
cp .env.example .env
```

Or create `.env` from scratch:

```
LLM_MODEL=ollama/llama3.2
LLM_TEMPERATURE=0.3
MAX_TOKENS=4000
ENABLE_HUMAN_IN_LOOP=false
PARALLEL_RESEARCH_DEPTH=1
```

---

## Running

```bash
python research_crew.py
```

You'll be prompted:

```
What should the crew research? How are startups using AI agents in 2026?
```

Type your topic and press Enter. The crew starts immediately.

**Expected run time:** 5–15 minutes depending on your machine and topic.

When finished you'll see:

```
Generated files:
   final_report.md — Complete research report
   summary.md      — One-page executive summary
```

---

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL` | `ollama/llama3.2` | Any Ollama model — see alternatives below |
| `LLM_TEMPERATURE` | `0.3` | Lower = more factual. Range: 0.0–1.0 |
| `MAX_TOKENS` | `4000` | Max tokens per agent response |
| `ENABLE_HUMAN_IN_LOOP` | `false` | Set `true` to give feedback mid-run |
| `PARALLEL_RESEARCH_DEPTH` | `1` | Research dimensions 1–6. Higher = more thorough, slower |

### Recommended models

| Model | Speed | Quality | Pull command |
|---|---|---|---|
| `ollama/llama3.2` | Fast | Good | `ollama pull llama3.2` |
| `ollama/llama3.1:8b` | Fast | Good | `ollama pull llama3.1:8b` |
| `ollama/mistral` | Medium | Good | `ollama pull mistral` |
| `ollama/llama3.3:70b` | Slow | Best | `ollama pull llama3.3:70b` |

---

## Research Depth

`PARALLEL_RESEARCH_DEPTH` controls how many angles the researcher investigates:

| Value | What gets researched |
|---|---|
| `1` | Current trends and data only *(fastest)* |
| `2` | + Key players and competitors |
| `3` | + Challenges and limitations |
| `4` | + Future predictions and opportunities |
| `5` | + Real-world case studies |
| `6` | + Costs and economics *(most thorough)* |

Start with `1`. Increase only if the report feels thin.

---

## Human-in-the-Loop Mode

Set `ENABLE_HUMAN_IN_LOOP=true` to be prompted at two points during the run:

1. **Mid-analysis** — the analyst asks you to clarify ambiguous findings
2. **Post-critique** — the critic asks for your feedback before the report is written

The terminal pauses and waits for your typed response each time.

---

## The Six Agents

| # | Agent | Job | Tools |
|---|---|---|---|
| 1 | **Researcher** | Searches the web, scrapes articles | WebSearch, DeepScrape, SaveNote |
| 2 | **Consolidator** | Groups raw findings into a structured summary | None |
| 3 | **Analyst** | Identifies patterns, quantifies insights, flags risks | HumanReview *(optional)* |
| 4 | **Critic** | Challenges assumptions, finds blind spots | HumanFeedback *(optional)* |
| 5 | **Report Writer** | Writes the full executive report | None |
| 6 | **Summary Writer** | Distils the report to one page | None |

Each agent runs exactly once, in order. No agent repeats.

---

## Troubleshooting

**`ollama: command not found`**
→ Install Ollama from [ollama.com](https://ollama.com) and restart your terminal.

**`connection refused` or `timeout`**
→ Run `ollama serve` in a separate terminal before starting the script.

**`model not found`**
→ Run `ollama pull llama3.2` (or whichever model is in your `.env`).

**Run takes over 20 minutes and seems stuck**
→ Press `Ctrl+C`. Then lower `PARALLEL_RESEARCH_DEPTH` to `1` and try a faster model like `ollama/llama3.2`.

**`ImportError: cannot import name 'DDGS'`**
→ Run `pip install ddgs` or `pip install duckduckgo-search`. The script tries both automatically.

**`final_report.md` is empty or missing**
→ The run was interrupted before the writer agents finished. Run again — earlier agents complete quickly on a retry.

---

## Project Structure

```
research-crew/
├── research_crew.py     # everything in one file
├── .env                 # your config (not committed to git)
├── .env.example         # template
├── README.md
├── final_report.md      # created after each run
└── summary.md           # created after each run
```

---

## `.env.example`

```
# Model — any model you have pulled in Ollama
LLM_MODEL=ollama/llama3.2

# 0.0 = deterministic, 1.0 = creative. 0.3 works well for research.
LLM_TEMPERATURE=0.3

# Max tokens per agent response. 4000 is safe for llama3.2.
MAX_TOKENS=4000

# Set true to give feedback during the run
ENABLE_HUMAN_IN_LOOP=false

# 1 = fast. 6 = thorough. Start with 1.
PARALLEL_RESEARCH_DEPTH=1
```

---

## License

MIT — free to use, modify, and distribute.