from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import get_config
from src.utils import load_json


@dataclass
class RetrievedContext:
    score: float
    entries: list[dict]


class KnowledgeRetriever:
    def __init__(self, knowledge_base_path: str | None = None) -> None:
        config = get_config()
        self.knowledge_base_path = knowledge_base_path or str(config.knowledge_base_path)
        self.entries = load_json(self.knowledge_base_path)
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        corpus = [
            " ".join(
                [
                    entry.get("intent", ""),
                    entry.get("title", ""),
                    entry.get("content", ""),
                    " ".join(entry.get("keywords", [])),
                ]
            )
            for entry in self.entries
        ]
        self.matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, intent: str | None = None, top_k: int | None = None) -> RetrievedContext:
        config = get_config()
        limit = top_k or config.top_k_context
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix)[0]
        ranked_indices = np.argsort(scores)[::-1]

        selected: list[dict] = []
        for idx in ranked_indices:
            entry = self.entries[idx]
            if intent and entry.get("intent") not in {intent, "General"}:
                continue
            candidate = dict(entry)
            candidate["score"] = float(scores[idx])
            selected.append(candidate)
            if len(selected) >= limit:
                break

        best_score = float(selected[0]["score"]) if selected else 0.0
        return RetrievedContext(score=best_score, entries=selected)
