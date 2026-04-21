# Database Design

MediChat uses SQLite for lightweight persistence. This keeps the project simple, portable, and easy to run in a student environment.

## Tables

### `sessions`

- `session_id` TEXT PRIMARY KEY
- `created_at` TEXT
- `updated_at` TEXT
- `language` TEXT

Purpose:

- identify a chat session
- track the latest session language
- support multi-turn conversation state

### `messages`

- `id` INTEGER PRIMARY KEY AUTOINCREMENT
- `session_id` TEXT
- `role` TEXT
- `original_text` TEXT
- `english_text` TEXT
- `language` TEXT
- `intent` TEXT
- `confidence` REAL
- `metadata_json` TEXT
- `created_at` TEXT

Purpose:

- store both user and assistant turns
- keep the original-language text and English processing text
- log predicted intent and confidence
- store retrieval metadata for analysis

## Why SQLite Was Chosen

- no separate server is required
- built into Python
- enough for demo-scale chat history
- easy to inspect during grading or development

## Optional Future Extensions

- add user identifiers
- store model version per response
- add analytics dashboards over the logged metadata
