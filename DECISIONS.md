# Decisions & Observations

## Phase 1: From notebook to project

### Structure
- Moved loading and chunking from the notebook into `src/ingest.py`: one loop
  over `data/*.txt` instead of hardcoded file paths.
- Read files with explicit UTF-8, because Windows' default encoding crashes
  on characters like `═`.
- Result: 120 paragraph chunks from 5 documents.

### Retrieval
- Switched ChromaDB from in-memory to `PersistentClient`, so the index is saved
  to `chroma_db/` and not rebuilt on every run.
- `build_index()` deletes and recreates the collection to avoid stale chunks.
- Top-K is a named constant (`TOP_K`) so it can be tuned in one place.

### Generation and prompting
- The API key is loaded from `.env` via python-dotenv and never committed.
- Temperature 0.2: RAG needs factual, consistent answers, not creativity.
- Experiment: a softer fallback ("ask me a question related to the company")
  let the model answer "capital of India" from general knowledge.
  Strict wording ("reply exactly: I don't know based on the provided
  documents.") refused correctly.
- Keep the fallback as a fixed phrase so evaluation code can detect refusals;
  a friendly message can be shown in the UI instead.
- Greetings ("hi man") got a refusal plus an irrelevant source. Added prompt
  rules for greetings and for citing sources only when the answer uses them.
- LLM output varies between runs even at temperature 0.2, so manual
  spot-checks aren't reliable. Phase 2 adds an automated evaluation set.

## Ideas to test in Phase 3
- Title-only chunks (e.g. the document header) carry no useful info. Filter them out?
- Section headings are dropped during chunking. Prepend the section name to each chunk?
- Add a similarity threshold so off-topic questions never reach the LLM
  (a code-level guard, since prompts aren't guarantees).