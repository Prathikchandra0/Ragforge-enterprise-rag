# Decisions & Observations

## Phase 1
- Moved loading and chunking from the notebook into `src/ingest.py`: one loop over `data/*.txt` instead of hardcoded files.
- Read files with explicit UTF-8, because Windows' default encoding crashes on characters like `═`.
- Result: 120 paragraph chunks from 5 documents.

## Ideas to test in Phase 3
- Title-only chunks (e.g. the document header) carry no useful info. Filter them out?
- Section headings are dropped during chunking. Prepend the section name to each chunk?