"""
Task 1: Dynamic Knowledge Base
Periodically updates a ChromaDB vector database from web sources and manual inputs.
"""

import hashlib
import datetime
import threading
import logging
import re
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


# ── ChromaDB setup with graceful fallback ────────────────────────────────────
def _make_chroma_client(persist_dir: str):
    try:
        import chromadb
        return chromadb.PersistentClient(path=persist_dir)
    except Exception as e:
        logger.warning(f"PersistentClient failed ({e}), using EphemeralClient")
        import chromadb
        return chromadb.EphemeralClient()


def _make_embedding_fn():
    try:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        return SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    except Exception as e:
        logger.warning(f"SentenceTransformer not available ({e}), using default embedding")
        try:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
            return DefaultEmbeddingFunction()
        except Exception:
            return None  # ChromaDB will use its built-in default


class KnowledgeBase:
    """
    Vector knowledge base backed by ChromaDB.
    Supports manual document ingestion, web scraping, and periodic background updates.
    """

    def __init__(self, persist_dir: str = "./data/chroma_db", collection_name: str = "nexus_kb"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.update_log: List[Dict] = []
        self.update_sources: List[str] = []
        self._scheduler_thread: Optional[threading.Thread] = None
        self._scheduler_running = False
        self._init_db()

    # ── Initialization ────────────────────────────────────────────────────────
    def _init_db(self):
        self.client = _make_chroma_client(self.persist_dir)
        ef = _make_embedding_fn()
        kwargs = {"name": self.collection_name, "metadata": {"hnsw:space": "cosine"}}
        if ef:
            kwargs["embedding_function"] = ef
        self.collection = self.client.get_or_create_collection(**kwargs)
        logger.info(f"KB ready – {self.collection.count()} chunks stored")

    # ── Document Management ───────────────────────────────────────────────────
    def add_document(self, text: str, metadata: Optional[Dict] = None,
                     doc_id: Optional[str] = None) -> Dict:
        """Chunk and add a document to the vector store."""
        result = {"success": False, "chunks": 0, "error": None}
        try:
            text = text.strip()
            if not text:
                result["error"] = "Empty document"
                return result

            doc_id = doc_id or hashlib.md5(text.encode()).hexdigest()
            chunks = self._chunk_text(text)
            base_meta = metadata or {}
            base_meta["added_at"] = datetime.datetime.now().isoformat()

            ids, documents, metadatas = [], [], []
            for i, chunk in enumerate(chunks):
                ids.append(f"{doc_id}_c{i}")
                documents.append(chunk)
                metadatas.append({**base_meta, "chunk": i, "total": len(chunks)})

            self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            result.update({"success": True, "chunks": len(chunks)})
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"add_document failed: {e}")
        return result

    def search(self, query: str, n_results: int = 4) -> List[Dict]:
        """Semantic search – returns ranked context passages."""
        try:
            total = self.collection.count()
            if total == 0:
                return []
            hits = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, total),
                include=["documents", "metadatas", "distances"],
            )
            results = []
            for doc, meta, dist in zip(
                hits["documents"][0], hits["metadatas"][0], hits["distances"][0]
            ):
                results.append({
                    "content": doc,
                    "metadata": meta,
                    "score": round(1.0 - dist, 3),
                })
            return results
        except Exception as e:
            logger.error(f"search failed: {e}")
            return []

    def delete_all(self) -> bool:
        try:
            self.client.delete_collection(self.collection_name)
            self._init_db()
            return True
        except Exception as e:
            logger.error(f"delete_all failed: {e}")
            return False

    def get_stats(self) -> Dict:
        return {
            "total_chunks": self.collection.count(),
            "sources": len(self.update_sources),
            "scheduler_running": self._scheduler_running,
            "recent_updates": self.update_log[-10:],
        }

    # ── Web Scraping ──────────────────────────────────────────────────────────
    def scrape_url(self, url: str, max_chars: int = 12_000) -> Optional[str]:
        try:
            headers = {"User-Agent": "NEXUS-Bot/1.0 (customer service AI research)"}
            resp = requests.get(url, headers=headers, timeout=20)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.content, "lxml")
            for tag in soup(["script", "style", "nav", "header", "footer", "aside", "iframe"]):
                tag.decompose()
            text = soup.get_text(separator=" ", strip=True)
            text = re.sub(r"\s+", " ", text).strip()
            return text[:max_chars]
        except Exception as e:
            logger.error(f"scrape_url({url}) failed: {e}")
            return None

    def update_from_url(self, url: str) -> Dict:
        entry = {
            "url": url,
            "timestamp": datetime.datetime.now().isoformat(),
            "success": False,
            "chunks": 0,
            "error": None,
        }
        text = self.scrape_url(url)
        if text:
            res = self.add_document(text, metadata={"source": url, "type": "web"})
            entry.update({"success": res["success"], "chunks": res["chunks"],
                          "error": res.get("error")})
        else:
            entry["error"] = "Scraping returned no content"
        self.update_log.append(entry)
        return entry

    # ── Scheduled Updates ─────────────────────────────────────────────────────
    def schedule_updates(self, urls: List[str], interval_hours: int = 24):
        """Launch a background thread that re-scrapes URLs every interval_hours."""
        import schedule as sch
        import time

        self.update_sources = urls

        def _job():
            logger.info(f"Scheduled KB update for {len(urls)} source(s)")
            for url in urls:
                self.update_from_url(url)

        def _runner():
            sch.every(interval_hours).hours.do(_job)
            _job()  # run once immediately
            while self._scheduler_running:
                sch.run_pending()
                time.sleep(60)

        if self._scheduler_thread and self._scheduler_thread.is_alive():
            self._scheduler_running = False
            self._scheduler_thread.join(timeout=5)

        self._scheduler_running = True
        self._scheduler_thread = threading.Thread(target=_runner, daemon=True)
        self._scheduler_thread.start()
        logger.info(f"KB scheduler started – every {interval_hours}h")

    def stop_scheduler(self):
        self._scheduler_running = False

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _chunk_text(text: str, size: int = 450, overlap: int = 50) -> List[str]:
        words = text.split()
        chunks, i = [], 0
        while i < len(words):
            chunks.append(" ".join(words[i: i + size]))
            i += size - overlap
        return chunks or [text]
