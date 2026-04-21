from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.database import ChatDatabase
from src.utils import compact_text


@dataclass
class DialogueContext:
    session_id: str
    history: list[dict[str, Any]]


class DialogueManager:
    def __init__(self, database: ChatDatabase | None = None) -> None:
        self.database = database or ChatDatabase()

    def get_context(self, session_id: str | None, language: str | None = None) -> DialogueContext:
        active_session = self.database.ensure_session(session_id, language=language)
        history = self.database.get_recent_messages(active_session, limit=8)
        return DialogueContext(session_id=active_session, history=history)

    def build_query(self, english_text: str, history: list[dict[str, Any]]) -> str:
        recent_user_turns = [
            item.get("english_text") or item.get("original_text", "")
            for item in history
            if item.get("role") == "user"
        ][-2:]
        if len(english_text.split()) >= 6 or not recent_user_turns:
            return english_text
        prior_context = " ".join(recent_user_turns)
        return compact_text(f"{prior_context} Follow-up question: {english_text}", max_chars=400)
