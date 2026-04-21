# Development Plan

## Goal

Upgrade the original intent-classification prototype into a final demonstrable system that combines classical NLP, multilingual handling, safe conversational response generation, and reproducible ML operations.

## Completed Upgrade Areas

1. Refactored the codebase into modular backend services.
2. Added a Gradio chat UI with session state and audio input.
3. Added SQLite persistence for conversations.
4. Added retrieval-based controlled generation.
5. Expanded the ML workflow to train and compare three required models.
6. Added DVC stages for prepare, train, and evaluate.

## Remaining Team Tasks

1. Replace or extend the current medical intent dataset with a larger real-world dataset if the team has time and approval.
2. Add more multilingual manual testing across target languages.
3. Expand the retrieval knowledge base with more medically reviewed educational snippets.
4. Record demo screenshots or a short demo video for presentation use.

## Recommended Demo Story

1. Show `dvc repro` to demonstrate reproducibility.
2. Show model comparison outputs in `documentation/model_comparison.md`.
3. Launch the app and ask an English text question.
4. Ask a follow-up question to demonstrate multi-turn handling.
5. Ask a non-English question to demonstrate translation behavior.
6. Use audio input to demonstrate speech-to-text.
7. Ask an unsupported question to show safe fallback behavior.

## Practical Scope Decisions

- Use Gradio instead of Streamlit to keep the chat UI simple and fast to demo.
- Keep SQLite lightweight instead of adding a server database.
- Use retrieval plus controlled generation instead of unrestricted LLM chat.
- Keep the baseline classifier as the main runtime model to preserve course alignment.
