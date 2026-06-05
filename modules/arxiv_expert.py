"""
Task 4: arXiv Research Expert using Groq (free)
"""

import logging
from typing import Dict, List, Optional
from groq import Groq

logger = logging.getLogger(__name__)


class ArxivExpert:
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.client = Groq(api_key=api_key)
        self.model  = model
        self._paper_cache: Dict[str, Dict] = {}
        try:
            import arxiv
            self._arxiv        = arxiv
            self._arxiv_client = arxiv.Client()
            self.available     = True
        except ImportError:
            logger.warning("arxiv package not installed.")
            self.available = False

    def _ask(self, prompt: str, max_tokens: int = 1000) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()

    def search_papers(self, query: str, max_results: int = 6,
                      category: Optional[str] = None) -> List[Dict]:
        if not self.available:
            return [{"error": "arxiv not installed", "title": "Unavailable", "abstract": ""}]
        try:
            full_query = f"({query}) AND cat:{category}" if category else query
            search     = self._arxiv.Search(query=full_query, max_results=max_results,
                                            sort_by=self._arxiv.SortCriterion.Relevance)
            papers = []
            for r in self._arxiv_client.results(search):
                p = {
                    "id":         r.entry_id.split("/")[-1],
                    "title":      r.title,
                    "authors":    [a.name for a in r.authors[:6]],
                    "abstract":   r.summary,
                    "published":  r.published.strftime("%Y-%m-%d") if r.published else "?",
                    "categories": r.categories,
                    "pdf_url":    r.pdf_url,
                    "entry_url":  r.entry_id,
                }
                papers.append(p)
                self._paper_cache[p["id"]] = p
            return papers
        except Exception as e:
            logger.error(f"arXiv search failed: {e}")
            return [{"error": str(e), "title": "Search failed", "abstract": str(e)}]

    def summarize_paper(self, paper: Dict) -> str:
        try:
            return self._ask(f"""Summarize this research paper for a technical audience.

Title: {paper['title']}
Authors: {', '.join(paper.get('authors', [])[:4])}
Published: {paper.get('published', '?')}
Abstract: {paper['abstract']}

Use these sections:
## 🎯 Core Contribution
## 🔍 Problem Addressed
## ⚙️ Key Methods
## 📊 Results & Impact
## 🔮 Significance""", max_tokens=900)
        except Exception as e:
            return f"⚠️ Summary failed: {e}"

    def explain_concept(self, concept: str, context_papers: Optional[List[Dict]] = None) -> str:
        ctx = ""
        if context_papers:
            ctx = "\n\nRelevant Papers:\n" + "".join(
                f"- {p['title']}: {p['abstract'][:200]}…\n" for p in context_papers[:3]
            )
        try:
            return self._ask(f"""Explain this concept clearly for a technical audience.

Concept: {concept}{ctx}

Use these sections:
## Definition
## Why It Matters
## How It Works
## Concrete Example
## Current Research Frontiers""", max_tokens=1200)
        except Exception as e:
            return f"⚠️ Explanation failed: {e}"

    def answer_with_papers(self, question: str, papers: List[Dict]) -> str:
        papers_text = "".join(
            f"\nPaper {i}: {p['title']}\n{p['abstract'][:500]}…\n---"
            for i, p in enumerate(papers[:5], 1)
        )
        try:
            return self._ask(f"""Answer this research question using the provided papers.

Question: {question}

Papers:
{papers_text}

- Answer directly in the first paragraph
- Cite papers by number (e.g. "According to Paper 2...")
- Synthesize across papers where relevant""", max_tokens=1400)
        except Exception as e:
            return f"⚠️ Q&A failed: {e}"

    def get_visualization_data(self, papers: List[Dict]) -> Dict:
        timeline = sorted(
            [{"date": p["published"], "title": p["title"][:50], "id": p["id"]}
             for p in papers if p.get("published") and p["published"] != "?"],
            key=lambda x: x["date"],
        )
        cat_counts: Dict[str, int] = {}
        for p in papers:
            for c in p.get("categories", []):
                top = c.split(".")[0]
                cat_counts[top] = cat_counts.get(top, 0) + 1
        return {"timeline": timeline, "category_counts": cat_counts}
