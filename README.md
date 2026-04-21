# MediChat

MediChat is a multilingual medical information chatbot built for the PROG8245 machine learning and NLP final project. The final version combines traditional ML intent classification, retrieval-based controlled response generation, multilingual handling, audio transcription, a Gradio web UI, SQLite chat logging, and DVC pipeline management.

## Core Features

- Traditional ML model comparison:
  - `TF-IDF + Multinomial Naive Bayes`
  - `TF-IDF + TruncatedSVD + Logistic Regression`
  - `TF-IDF + dense PCA + Logistic Regression`
- Chat-style Gradio interface with multi-turn conversation
- Text and audio input
- Language detection plus translate-to-English processing flow
- Controlled generation grounded in retrieved medical context
- Confidence and retrieval thresholding for out-of-scope fallback
- SQLite-based session and message logging
- DVC-ready prepare, train, and evaluate pipeline

## Updated Architecture

```text
User text/audio
  -> speech-to-text (optional, OpenAI)
  -> language detection
  -> translation to English when needed
  -> dialogue manager builds contextual query
  -> baseline ML classifier predicts intent
  -> TF-IDF retrieval over curated medical knowledge base
  -> controlled response generation from allowed context only
  -> translate response back to user language when needed
  -> persist turn in SQLite
  -> display in Gradio chat UI
```

## Project Structure

```text
PROG8245_Final_Project/
├── app.py
├── data/
│   ├── raw/
│   │   ├── medical_intent_dataset.csv
│   │   ├── medical_knowledge_base.json
│   │   └── intent_responses.json
│   ├── processed/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── dataset_summary.json
│   └── medichat.sqlite3
├── documentation/
│   ├── architecture.md
│   ├── development_plan.md
│   ├── user_manual.md
│   ├── model_comparison.md
│   └── database_design.md
├── models/
│   ├── baseline_nb.joblib
│   ├── svd_logreg.joblib
│   ├── pca_logreg.joblib
│   └── *.metadata.json
├── src/
│   ├── config.py
│   ├── data_processing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── dialogue_manager.py
│   ├── retrieval.py
│   ├── translation.py
│   ├── speech_to_text.py
│   ├── response_generator.py
│   ├── database.py
│   └── utils.py
├── dvc.yaml
├── params.yaml
├── requirements.txt
└── .env.example
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add `OPENAI_API_KEY` to `.env` if you want audio transcription and OpenAI-based translation/response phrasing. The project still runs without the key, but non-English translation and audio transcription will gracefully degrade.

## Reproducible ML Pipeline

Run the full pipeline:

```bash
dvc repro
```

Or run the stages directly:

```bash
python -m src.data_processing
python -m src.train
python -m src.evaluate
```

Outputs:

- processed train/test data in `data/processed/`
- trained models in `models/`
- evaluation JSON, CSV, markdown, and confusion matrices in `documentation/`

## Run the Web App

Train the models first, then launch:

```bash
python app.py
```

The UI supports:

- text questions
- uploaded audio or microphone audio
- multilingual flow
- multi-turn history
- turn-by-turn metadata display

## Controlled Generation Design

MediChat does not generate unrestricted medical advice. Each turn follows this controlled workflow:

1. Detect language and translate to English when needed.
2. Classify the intent with a traditional ML model.
3. Retrieve supporting snippets from a small curated knowledge base.
4. Reject low-confidence or low-relevance queries with a safe fallback.
5. Generate the final answer using only retrieved context.

This keeps the project aligned with the course requirement for explainable ML while still making the demo conversational.

## Safety Scope

- MediChat is not a diagnosis system.
- It provides general educational information only.
- It must not be used for emergencies, medication decisions, or personalized treatment.
- Severe or worsening symptoms should be referred to licensed medical professionals.

## Notes on the Dataset

The repository currently includes a balanced respiratory-health intent dataset in `data/raw/medical_intent_dataset.csv`, which keeps the project lightweight and reproducible for a student environment. The training code is organized so the dataset can be swapped for a larger medical intent classification CSV later if the team decides to extend the scope.
