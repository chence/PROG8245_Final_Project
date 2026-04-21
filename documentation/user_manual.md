# User Manual

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Optional:

- Add `OPENAI_API_KEY` to `.env` for speech-to-text and better translation.

## Train and Evaluate

Run the reproducible pipeline:

```bash
dvc repro
```

Or run manually:

```bash
python -m src.data_processing
python -m src.train
python -m src.evaluate
```

## Launch the Chatbot

```bash
python app.py
```

Open the local Gradio link in your browser.

## How to Use the App

1. Type a general medical information question and click `Send`.
2. Or record/upload audio and click `Transcribe and Send`.
3. Review the chatbot answer in the chat window.
4. Check the `Turn details` panel for:
   - detected language
   - predicted intent
   - classifier confidence
   - retrieval score
   - whether the answer was treated as supported

## Multilingual Behavior

- English input returns English only.
- Non-English input is processed in English internally.
- The user receives:
  - an answer in the original language
  - an English translation shown in the same response

If OpenAI translation is unavailable, non-English messages may not be translated correctly.

## Example Questions

- `I have a sore throat and mild fever.`
- `What should I do at home for a cough?`
- `Should I see a doctor if breathing feels harder?`
- `Can I combine cold medicine with pain reliever?`
- `Tengo tos y fiebre leve, que hago?`

## Limitations

- The medical scope is intentionally narrow and educational.
- The project does not diagnose illness.
- Translation and speech features depend on API availability.
- The retrieval knowledge base is curated and relatively small.
