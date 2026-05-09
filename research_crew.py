"""
Multi-Agent Research Crew
- 6 Agents: Researcher, Consolidator, Analyst, Devil's Advocate, Report Writer, Summary Writer
- One agent per task — no agent ever runs twice
- Blocked-domain list prevents Forbes/paywalled hallucination loops
- Hard-stop cache prevents infinite retry on bad URLs
- Silent save_note on malformed JSON — never retries
- Deep web scraping for full article content
- Human-in-the-loop for quality control
- .env configuration for all API keys

100% FREE - Uses Ollama + DuckDuckGo Search
"""

import os
import re
import json
import traceback
from typing import List
from datetime import datetime
from pathlib import Path

import time
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
try:
    from ddgs import DDGS                      # ddgs >= 6.x
except ImportError:
    from duckduckgo_search import DDGS         # duckduckgo-search <= 5.x
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

# ============= CONFIGURATION =============
load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "ollama/llama3.2")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4000"))
ENABLE_HUMAN_IN_LOOP = os.getenv("ENABLE_HUMAN_IN_LOOP", "false").lower() == "true"
PARALLEL_DEPTH = int(os.getenv("PARALLEL_RESEARCH_DEPTH", "1"))


# ── per-run deduplication caches ────────────────────────────────────────────
_search_cache: dict = {}
_scrape_cache: dict = {}

# ── FIX 7: domains the LLM hallucinates from training memory ────────────────
# These are paywalled, membership-only, or chronically 404 for bots.
# Any URL containing these strings is rejected BEFORE a network call is made.
_BLOCKED_DOMAINS = {
    "forbes.com",
    "forbestechcouncil.com",
    "wsj.com",
    "nytimes.com",
    "ft.com",
    "bloomberg.com",
    "hbr.org",
    "sciencedirect.com",
    "springer.com",
    "researchgate.net",
    "academia.edu",
}

def _is_blocked_url(url: str) -> bool:
    url_lower = url.lower()
    return any(domain in url_lower for domain in _BLOCKED_DOMAINS)


def make_llm(temperature: float = LLM_TEMPERATURE) -> LLM:
    return LLM(
        model=LLM_MODEL,
        temperature=temperature,
        max_tokens=MAX_TOKENS,
        timeout=300,
    )


llm = make_llm()


# ============= HELPERS =============

def _is_junk_content(text: str) -> bool:
    """Return True when scraped text is a 404 page, redirect, or navigation dump."""
    if not text or len(text.strip()) < 150:
        return True
    junk_signals = [
        "404", "page not found", "can't find the page",
        "nothing was found at this location",
        "moved permanently", "redirecting",
    ]
    snippet = text.lower()[:300]
    return sum(1 for s in junk_signals if s in snippet) >= 2


def _safe_parse_json(raw: str) -> dict:
    """Parse JSON tolerantly — single quotes, trailing commas, bare text."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    fixed = re.sub(r"'([^']*)'", r'"\1"', raw)
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    return {"title": "Research Note", "content": raw}


# ============= TOOLS =============

@tool("WebSearch")
def web_search_tool(query: str) -> str:
    """
    Search the web using DuckDuckGo (free, no API key).
    Input: A search query string — use a DIFFERENT query each call.
    Output: Formatted results with titles, links, and snippets.
    Do NOT call this with the same query twice; use the cached result instead.
    """
    cache_key = query.strip().lower()
    if cache_key in _search_cache:
        # FIX 8: hard-stop message — LLM sees "already done" and moves on
        return f"[ALREADY SEARCHED — do not search this again]: {_search_cache[cache_key][:500]}"

    for attempt in range(4):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))
                formatted = [
                    f"{i}. {r['title']}\n   Link: {r['href']}\n   {r['body']}\n"
                    for i, r in enumerate(results, 1)
                ]
                output = "\n".join(formatted) if formatted else "No results found."
                _search_cache[cache_key] = output
                return output
        except Exception as e:
            if "ratelimit" in str(e).lower() or "202" in str(e):
                time.sleep(5 * (attempt + 1))
            else:
                return f"Search error: {str(e)}. Use a different query."
    return "Search failed after retries. Move on with existing findings."


def scrape_website(url: str, max_length: int = 2500) -> str:
    """Scrape and clean text content from a URL."""
    # FIX 7: block known hallucinated/paywalled domains before any network call
    if _is_blocked_url(url):
        return f"BLOCKED: {url} is a paywalled or membership site — do not retry this URL."
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = requests.get(url, timeout=10, headers=headers)
        if resp.status_code >= 400:
            return f"HTTP {resp.status_code}: page not available — skip this URL."
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
            tag.decompose()
        main = (
            soup.find("article")
            or soup.find("main")
            or soup.find("div", class_="content")
            or soup
        )
        lines = (line.strip() for line in main.get_text().splitlines())
        clean = " ".join(phrase for line in lines for phrase in line.split("  ")).strip()
        if _is_junk_content(clean):
            return f"Junk content at {url} — skip this URL."
        return clean[:max_length] + "..." if len(clean) > max_length else clean
    except Exception as e:
        return f"Scraping failed: {str(e)} — skip this URL."


@tool("DeepScrape")
def deep_scrape_tool(query_with_url: str) -> str:
    """
    Search then scrape full article content.
    Input format: "search query" OR "search query | https://specific-url.com"
    RULES:
      - Only pass URLs that appeared in the CURRENT session's search results.
      - Never pass forbes.com, wsj.com, nytimes.com, or other paywalled sites.
      - If a URL is blocked or fails, the tool automatically falls back to a fresh search.
      - Do NOT call with the same arguments twice.
    """
    parts = query_with_url.split(" | ")
    search_query = parts[0].strip()
    specific_url = parts[1].strip() if len(parts) > 1 else None

    # FIX 8: hard-stop on repeated identical calls
    cache_key = query_with_url.strip().lower()
    if cache_key in _scrape_cache:
        return f"[ALREADY SCRAPED — do not call this again. Use existing content.]"

    try:
        if specific_url:
            # FIX 7: block before network call; clear URL so we fall through to search
            if _is_blocked_url(specific_url):
                specific_url = None
            else:
                result = scrape_website(specific_url)
                if any(w in result for w in ["BLOCKED", "HTTP ", "Junk content", "Scraping failed"]):
                    specific_url = None   # fall through to search
                else:
                    _scrape_cache[cache_key] = result
                    return result

        # No usable URL — do a fresh DuckDuckGo search
        with DDGS() as ddgs:
            results = list(ddgs.text(search_query, max_results=5))

        scraped_parts = []
        for r in results[:4]:
            url = r["href"]
            if _is_blocked_url(url):
                continue
            content = scrape_website(url)
            if any(w in content for w in ["BLOCKED", "HTTP ", "Junk content", "Scraping failed"]):
                continue
            scraped_parts.append(f"### {r['title']}\nURL: {url}\n{content}\n")
            if len(scraped_parts) >= 2:
                break

        output = "\n".join(scraped_parts) if scraped_parts else (
            "No scrapeable content found for this query. "
            "Use WebSearch results directly and move on."
        )
        _scrape_cache[cache_key] = output
        return output

    except Exception as e:
        return f"Deep scrape error: {str(e)}. Use WebSearch results directly."


@tool("SaveResearchNote")
def save_note_tool(content_json: str) -> str:
    """
    Append a research finding to research_notes.md.
    Input: JSON string — {"title": "Short title", "content": "Your finding"}
    On any parse failure the note is saved as plain text — never errors out.
    """
    # FIX 9: never return an error message — silently salvage whatever was passed.
    # An error response causes the LLM to retry in a loop; "OK" lets it move on.
    if not content_json or content_json.strip() in ("{", "}", "", "null"):
        # Nothing usable — silently skip rather than erroring
        return "Note skipped (empty input). Continue with research."
    try:
        data = _safe_parse_json(content_json)
        title = data.get("title") or "Untitled"
        content = data.get("content") or str(data)
    except Exception:
        title = "Research Note"
        content = content_json.strip()

    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("research_notes.md", "a", encoding="utf-8") as f:
            f.write(f"\n## {title} (Saved: {timestamp})\n{content}\n---\n")
        return f"Note '{title}' saved. Continue research."
    except Exception as e:
        return f"Could not write file ({e}). Continue research."


@tool("RequestHumanReview")
def human_review_tool(question: str) -> str:
    """Pause and ask the human operator a question before proceeding."""
    if not ENABLE_HUMAN_IN_LOOP:
        return "Human review disabled. Proceed with current information."
    print(f"\nAGENT REQUEST: {question}\n" + "-" * 50)
    response = input("Human response: ")
    print("-" * 50)
    return response


@tool("RequestHumanFeedback")
def human_feedback_tool(report_section: str) -> str:
    """Ask the human operator for feedback on a draft section before finalizing."""
    if not ENABLE_HUMAN_IN_LOOP:
        return "Human feedback disabled. Proceed automatically."
    print(f"\nAGENT SEEKING FEEDBACK ON:\n{report_section[:500]}\n" + "-" * 50)
    response = input("Your feedback: ")
    print("-" * 50)
    return response


# ============= AGENTS =============

researcher = Agent(
    role="Senior Research Analyst",
    goal="Find comprehensive, accurate information from multiple web sources",
    backstory="""You are an expert researcher who always cites sources.
    STRICT RULES:
    - Call WebSearch with a DIFFERENT query each time — never repeat a query.
    - Only pass URLs from THIS session's search results to DeepScrape.
    - Never use forbes.com, wsj.com, nytimes.com, sciencedirect.com, or any paywalled site.
    - If DeepScrape or WebSearch returns [ALREADY ...], move on — do not call again.
    - Save findings with SaveResearchNote using valid JSON format.
    - Stop researching once you have 3-5 solid findings with sources.""",
    tools=[web_search_tool, deep_scrape_tool, save_note_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=12,
)

consolidator = Agent(
    role="Research Consolidator",
    goal="Organise raw research findings into a single structured master summary",
    backstory="""You are a meticulous editor. You take research results and group them
    by theme into one clean markdown document. You list every source link, surface
    contradictions, and note gaps. You do NOT interpret or make recommendations.
    One document, then stop.""",
    tools=[],
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=4,
)

analyst = Agent(
    role="Strategic Data Analyst",
    goal="Produce a single strategic analysis document from the consolidated research summary",
    backstory="""You turn a clean research summary into strategic insights.
    You identify hype vs. reality, quantify findings, and map agreement vs. disagreement.
    You produce ONE analysis document and stop.""",
    tools=[human_review_tool] if ENABLE_HUMAN_IN_LOOP else [],
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=6,
)

devils_advocate = Agent(
    role="Devil's Advocate & Critic",
    goal="Deliver a focused critique of the strategic analysis",
    backstory="""Your job: find what others missed. Challenge sources, methodology, conclusions.
    OUTPUT RULE: Write ONLY your own critique as numbered points with improvements.
    Never copy, quote, or repeat the analysis text. Never include RequestHumanFeedback
    blocks in your written output — call the tool if needed, then write your critique.""",
    tools=[human_feedback_tool] if ENABLE_HUMAN_IN_LOOP else [],
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=6,
)

report_writer = Agent(
    role="Executive Report Writer",
    goal="Synthesize analysis and critique into one complete balanced executive report",
    backstory="""You write for busy executives — every paragraph has a point.
    Synthesize in your own words. Never paste other agents' outputs verbatim.
    Produce ONE report in the required structure and stop.""",
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=8,
)

summary_writer = Agent(
    role="Executive Summary Writer",
    goal="Distil the full report into a tight one-page executive summary",
    backstory="""You write for C-suite readers with 60 seconds. Extract only:
    core problem/opportunity (1 sentence), top 3 findings, top 3 recommendations,
    biggest risk. Nothing else — no methodology, no sources, no critique text.""",
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=4,
)


# ============= TASK BUILDERS =============

RESEARCH_DIMENSIONS = [
    "CURRENT TRENDS AND DATA (2025-2026 only). Find specific numbers, statistics, concrete examples. Use WebSearch multiple times with DIFFERENT queries.",
    "KEY PLAYERS AND COMPETITORS. Main companies, tools, or people involved. Use DeepScrape for their official pages (skip paywalled sites).",
    "CHALLENGES AND LIMITATIONS. What doesn't work? Trade-offs? Use WebSearch with different queries.",
    "FUTURE PREDICTIONS AND OPPORTUNITIES. What's coming next? Expert predictions.",
    "REAL-WORLD EXAMPLES AND CASE STUDIES. Specific companies succeeding or failing.",
    "COSTS AND ECONOMICS. Financial implications, ROI, pricing models.",
]


def create_parallel_research_tasks(topic: str) -> List[Task]:
    dims = RESEARCH_DIMENSIONS[:PARALLEL_DEPTH]
    return [
        Task(
            description=f"""Research dimension {i+1} of {len(dims)}: {topic} — {dim}

            MANDATORY RULES:
            1. Call WebSearch at least 2 times with DIFFERENT queries.
            2. Only use URLs from search results — never hardcoded or memorised URLs.
            3. Never scrape: forbes.com, wsj.com, nytimes.com, sciencedirect.com, or paywalled sites.
            4. If a tool returns [ALREADY ...] — stop calling it and move on.
            5. Save 1-2 key findings with SaveResearchNote:
               {{"title": "Short title", "content": "Your finding with source URL"}}
            6. Once you have 3+ good findings, write your final answer immediately.""",
            expected_output=f"Dimension {i+1} findings: 3-5 data points with source URLs",
            agent=researcher,
        )
        for i, dim in enumerate(dims)
    ]


def create_research_crew(topic: str) -> Crew:
    """
    Agent → Task (one-to-one — no agent runs twice):
        researcher      → research_tasks
        consolidator    → consolidation_task
        analyst         → analysis_task
        devils_advocate → critique_task
        report_writer   → report_task
        summary_writer  → summary_task
    """
    _search_cache.clear()
    _scrape_cache.clear()

    research_tasks = create_parallel_research_tasks(topic)

    consolidation_task = Task(
        description=f"""Consolidate research findings about: {topic}

        Synthesize into a MASTER RESEARCH SUMMARY (markdown):
        1. Key findings by theme (with sources)
        2. Data points and statistics
        3. Source links by category
        4. Where findings agree or contradict
        5. Notable gaps

        Output ONLY the summary. No analysis, no recommendations.""",
        expected_output="Structured markdown research summary with sources",
        agent=consolidator,
        context=research_tasks,
    )

    analysis_task = Task(
        description=f"""Strategic analysis of: {topic}

        From the consolidated summary only, identify:
        1. The 3-5 most significant patterns (with evidence)
        2. Where sources agree vs. disagree
        3. Critical blind spots
        4. Biggest opportunity or risk (justified)
        5. Key decision metrics
        6. Past 12 months vs. future predictions

        Be specific. Quantify when possible.
        Output ONLY your analysis — do not repeat the raw research.""",
        expected_output="Strategic analysis with quantified insights and risk/opportunity assessment",
        agent=analyst,
        context=[consolidation_task],
    )

    critique_task = Task(
        description=f"""Critique the strategic analysis of: {topic}

        Output ONLY numbered weaknesses with specific improvements. Do NOT copy analysis text.

        Cover:
        1. Incorrect or incomplete assumptions
        2. Missed counter-evidence or perspectives
        3. Ignored economic/technical/social factors
        4. How recommendations could backfire
        5. What a disagreeing expert would say
        6. Source credibility and bias
        7. Whether conclusions fit the evidence""",
        expected_output="Numbered critique: each weakness + specific improvement",
        agent=devils_advocate,
        context=[analysis_task],
    )

    report_task = Task(
        description=f"""Final executive report on: {topic}

        Synthesize analysis + critique IN YOUR OWN WORDS. Do not paste verbatim.

        ## Executive Summary
        ## Key Findings (with confidence: High/Medium/Low)
        ## Detailed Analysis
        ### What's Working
        ### What's Not Working
        ### Who's Winning and Why
        ## Contradictions & Debates
        ## Recommendations
        ### Immediate Actions (Next 30 days)
        ### Strategic Moves (Next 3-6 months)
        ### Long-term Considerations (6+ months)
        ## Risk Assessment
        ## Open Questions & Gaps
        ## Sources
        ## Methodology

        End with "END OF REPORT" on its own line.""",
        expected_output="Complete professional markdown report ending with END OF REPORT",
        agent=report_writer,
        context=[analysis_task, critique_task],
        output_file="final_report.md",
    )

    summary_task = Task(
        description=f"""One-page executive summary for: {topic}

        Extract from the full report:
        - Problem/opportunity (1 sentence)
        - Top 3 findings (bullets, ≤20 words each)
        - Top 3 recommendations (numbered, ≤20 words each)
        - Biggest risk (1 sentence)

        Output ONLY these four sections. Nothing else.""",
        expected_output="One-page executive summary with exactly 4 sections",
        agent=summary_writer,
        context=[report_task],
        output_file="summary.md",
    )

    all_tasks = research_tasks + [
        consolidation_task, analysis_task, critique_task, report_task, summary_task
    ]

    return Crew(
        agents=[researcher, consolidator, analyst, devils_advocate, report_writer, summary_writer],
        tasks=all_tasks,
        process=Process.sequential,
        verbose=True,
    )


# ============= ENTRY POINT =============

def main():
    print("\n" + "=" * 60)
    print("MULTI-AGENT RESEARCH SYSTEM")
    print("=" * 60)
    print(f"   Model         : {LLM_MODEL}")
    print(f"   Parallel depth: {PARALLEL_DEPTH} research dimension(s)")
    print(f"   Human review  : {'Enabled' if ENABLE_HUMAN_IN_LOOP else 'Disabled'}")

    topic = input("\nWhat should the crew research? ").strip()
    if not topic:
        topic = "The current state of AI-powered content creation tools for solo creators in 2026"
        print(f"Using example topic: {topic}")

    try:
        crew = create_research_crew(topic)
        print(f"\n{'=' * 60}\nStarting research on: {topic}\n{'=' * 60}\n")
        crew.kickoff()
    except KeyboardInterrupt:
        print("\nResearch interrupted. Partial results may have been saved.")
        return
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("   1. Make sure Ollama is running: ollama serve")
        print("   2. Verify your internet connection")
        print("   3. Check the model is pulled: ollama list")
        print("   4. Pull the model if missing: ollama pull llama3.2")
        traceback.print_exc()
        return

    print("\nGenerated files:")
    for fname, label in [
        ("final_report.md",   "Complete research report"),
        ("summary.md",        "One-page executive summary"),
    ]:
        if os.path.exists(fname):
            print(f"   {fname} — {label}")

    if os.path.exists("summary.md"):
        content = Path("summary.md").read_text()
        print("\nEXECUTIVE SUMMARY PREVIEW:\n" + "-" * 40)
        print(content[:500] + "..." if len(content) > 500 else content)
        print("-" * 40)

    print("\nResearch complete!")


if __name__ == "__main__":
    main()