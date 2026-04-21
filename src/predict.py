from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.config import get_config
from src.dialogue_manager import DialogueManager
from src.response_generator import SAFETY_NOTICE, generate_controlled_response
from src.retrieval import KnowledgeRetriever
from src.translation import TranslationResult, detect_language, translate_text
from src.utils import load_json

UNSUPPORTED_MESSAGE = (
    "I can only answer supported, general medical information questions in this course project. "
    "Please rephrase the question or ask a healthcare professional for personalized advice."
)


@dataclass
class PredictionArtifacts:
    model: Any
    responses: dict[str, str]
    retriever: KnowledgeRetriever
    dialogue_manager: DialogueManager


class MediChatEngine:
    def __init__(self, model_name: str = "baseline_nb") -> None:
        config = get_config()
        model_path = config.model_artifact_path(model_name)
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model artifact not found at {model_path}. Run the training pipeline before launching the app."
            )
        self.config = config
        self.artifacts = PredictionArtifacts(
            model=joblib.load(model_path),
            responses=load_json(config.responses_path),
            retriever=KnowledgeRetriever(str(config.knowledge_base_path)),
            dialogue_manager=DialogueManager(),
        )

    def classify(self, english_text: str) -> tuple[str, float]:
        probabilities = self.artifacts.model.predict_proba([english_text])[0]
        best_index = int(np.argmax(probabilities))
        return str(self.artifacts.model.classes_[best_index]), float(probabilities[best_index])

    def process_message(
        self,
        user_text: str,
        *,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        source_language = detect_language(user_text)
        to_english: TranslationResult
        if source_language == "en":
            to_english = TranslationResult(
                text=user_text.strip(),
                translated=False,
                source_language="en",
                target_language="en",
                provider="local",
            )
        else:
            to_english = translate_text(user_text, "en", source_language=source_language)

        context = self.artifacts.dialogue_manager.get_context(session_id, language=source_language)
        contextual_query = self.artifacts.dialogue_manager.build_query(to_english.text, context.history)
        intent, confidence = self.classify(contextual_query)
        retrieved = self.artifacts.retriever.retrieve(contextual_query, intent=intent)

        supported = confidence >= self.config.confidence_threshold and retrieved.score >= self.config.retrieval_threshold
        if supported:
            fallback_message = self.artifacts.responses.get(intent, UNSUPPORTED_MESSAGE)
            english_response = generate_controlled_response(
                intent=intent,
                user_question=to_english.text,
                context_items=retrieved.entries,
                conversation_history=context.history,
                fallback_message=fallback_message,
            )
        else:
            intent = "unsupported"
            english_response = f"{UNSUPPORTED_MESSAGE}\n\n{SAFETY_NOTICE}"

        translated_response = (
            translate_text(english_response, source_language, source_language="en").text
            if source_language != "en"
            else english_response
        )

        self.artifacts.dialogue_manager.database.log_message(
            context.session_id,
            role="user",
            original_text=user_text,
            english_text=to_english.text,
            language=source_language,
            metadata={"contextual_query": contextual_query},
        )
        self.artifacts.dialogue_manager.database.log_message(
            context.session_id,
            role="assistant",
            original_text=translated_response,
            english_text=english_response,
            language=source_language,
            intent=intent,
            confidence=confidence,
            metadata={
                "retrieval_score": retrieved.score,
                "retrieved_titles": [entry["title"] for entry in retrieved.entries],
                "supported": supported,
            },
        )

        return {
            "session_id": context.session_id,
            "language": source_language,
            "english_text": to_english.text,
            "intent": intent,
            "confidence": confidence,
            "retrieval_score": retrieved.score,
            "supported": supported,
            "response": translated_response,
            "english_response": english_response,
            "english_translation": english_response if source_language != "en" else "",
            "retrieved_context": retrieved.entries,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single MediChat prediction.")
    parser.add_argument("text")
    parser.add_argument("--model-name", default="baseline_nb")
    args = parser.parse_args()
    engine = MediChatEngine(model_name=args.model_name)
    print(engine.process_message(args.text))
