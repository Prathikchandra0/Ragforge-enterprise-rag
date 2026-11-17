import json
import sys
from pathlib import Path

# Let this script import modules from the src/ folder
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

from retriever import retrieve

QUESTIONS_FILE = ROOT / "eval" / "questions.json"
K = 5


def first_hit_rank(chunks, expected_snippets):
    """Position (1, 2, 3...) of the first chunk containing an expected snippet, or None."""
    for rank, chunk in enumerate(chunks, start=1):
        if any(snippet in chunk["text"] for snippet in expected_snippets):
            return rank
    return None


def evaluate_retrieval(k=K):
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    answerable = [q for q in questions if q["answerable"]]

    hits = 0
    reciprocal_ranks = []
    misses = []

    for q in answerable:
        chunks = retrieve(q["question"], n_results=k)
        sources = [c["source"] for c in chunks]
        rank = first_hit_rank(chunks, q["expected_snippets"])

        if rank:
            hits += 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)
            misses.append((q, sources))

    hit_rate = hits / len(answerable)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    return hit_rate, mrr, misses


if __name__ == "__main__":
    for k in (1, 3, 5):
        hit_rate, mrr, misses = evaluate_retrieval(k)
        missed_ids = [q["id"] for q, _ in misses]
        print(f"K={k}:  hit rate {hit_rate:.0%}   MRR {mrr:.2f}   missed {missed_ids}")