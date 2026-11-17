import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

from rag import answer, client, MODEL

QUESTIONS_FILE = ROOT / "eval" / "questions.json"
RESULTS_FILE = ROOT / "eval" / "answer_results.json"
REFUSAL = "I don't know based on the provided documents."

JUDGE_PROMPT = """You are grading an AI assistant's answer.

Question: {question}
Expected answer (key facts): {expected}
Assistant's answer: {reply}

Does the assistant's answer contain the key facts of the expected answer
without contradicting them? Extra correct details are fine.
Reply with exactly one word: CORRECT or INCORRECT."""


def is_refusal(reply):
    return REFUSAL.lower() in reply.lower()


def judge(question, expected, reply):
    """Ask the LLM whether the reply matches the expected answer."""
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[{
            "role": "user",
            "content": JUDGE_PROMPT.format(question=question, expected=expected, reply=reply),
        }],
    )
    verdict = response.choices[0].message.content.strip().upper()
    return verdict.startswith("CORRECT")


def evaluate_answers():
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    results = []

    for q in questions:
        reply, _ = answer(q["question"])

        if q["answerable"]:
            passed = not is_refusal(reply) and judge(q["question"], q["expected_answer"], reply)
        else:
            passed = is_refusal(reply)

        results.append({
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "reply": reply,
            "passed": passed,
        })
        print(f"{'PASS' if passed else 'FAIL'}  #{q['id']}  {q['question']}")
        time.sleep(1)  # stay under Groq's rate limit

    return results


def score(results, category=None):
    subset = [r for r in results if category is None or r["category"] == category]
    return sum(r["passed"] for r in subset) / len(subset)


if __name__ == "__main__":
    results = evaluate_answers()
    RESULTS_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nOverall:       {score(results):.0%}")
    for category in ("fact", "paraphrased", "unanswerable"):
        print(f"{category:<14} {score(results, category):.0%}")
    print(f"\nFull replies saved to {RESULTS_FILE.name}")
    