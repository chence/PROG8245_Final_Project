# Architecture

## System Overview

MediChat uses a layered architecture so each course requirement stays visible and explainable.

```text
Text / Audio Input
    ->
Gradio Web UI
    ->
Speech-to-Text Module
    ->
Language Detection + Translation Layer
    ->
Dialogue Manager
    ->
Intent Classification Model
    ->
Knowledge Retrieval
    ->
Controlled Response Generator
    ->
Translation Back to User Language
    ->
SQLite Logging + Chat Display
```

## Main Modules

### `src/config.py`
Centralizes paths, thresholds, model names, and environment-based settings.

### `src/data_processing.py`
Loads the raw dataset, cleans text, creates stratified train/test splits, and saves reproducible processed files for DVC.

### `src/train.py`
Trains the three required classical ML pipelines:

- `baseline_nb`
- `svd_logreg`
- `pca_logreg`

### `src/evaluate.py`
Evaluates all trained models on the same test split and saves:

- JSON metrics
- CSV comparison table
- Markdown comparison table
- one confusion matrix image per model

### `src/translation.py`
Handles language detection locally with `langdetect` and uses OpenAI translation only when an API key is configured.

### `src/speech_to_text.py`
Uses OpenAI audio transcription for uploaded or microphone-recorded speech.

### `src/dialogue_manager.py`
Builds lightweight multi-turn context and turns short follow-up questions into more contextual classifier queries.

### `src/retrieval.py`
Uses TF-IDF similarity over a curated medical knowledge base so response generation is grounded in controlled context.

### `src/response_generator.py`
Creates the final answer. If OpenAI is available, the prompt explicitly limits the model to the retrieved context and forbids diagnosis. If OpenAI is unavailable, a deterministic local response is generated from the same retrieved snippets.

### `src/database.py`
Stores session metadata and turn history in SQLite for demo persistence and simple analytics.

### `src/predict.py`
Coordinates the full runtime inference flow used by the web application.

## Runtime Flow

1. User enters text or audio.
2. Audio is transcribed if provided.
3. Input language is detected.
4. Non-English text is translated to English for processing.
5. Recent conversation history is used to enrich short follow-up questions.
6. The baseline classifier predicts intent and confidence.
7. The retriever selects top medical knowledge snippets for that intent.
8. If confidence or retrieval score is too low, the chatbot returns a safe unsupported-message fallback.
9. Otherwise, the response generator produces a short conversational answer using only allowed context.
10. The answer is translated back when needed and stored in SQLite.

## Why This Fits the Course

- Keeps traditional ML front and center for classification and evaluation
- Adds multilingual and audio layers without replacing the classical pipeline
- Uses explainable retrieval rather than unrestricted generation
- Supports reproducible experimentation through DVC
- Remains small enough to be understandable for a class project
