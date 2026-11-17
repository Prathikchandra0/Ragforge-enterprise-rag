import os

from dotenv import load_dotenv
from openai import OpenAI

from retriever import retrieve

load_dotenv()  # reads GROQ_API_KEY from the .env file

MODEL = "openai/gpt-oss-20b"
TEMPERATURE = 0.2

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file.")

client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

SYSTEM_PROMPT = """You are NovaTech Solutions' internal knowledge assistant.

Rules:
1. Answer ONLY using information in the CONTEXT. Never use general knowledge,
   even for well-known facts.
2. If the CONTEXT does not contain the answer, reply exactly:
   I don't know based on the provided documents.
3. If the user only greets you, greet them briefly and ask what company
   question they have.
4. When you answer from the CONTEXT, end with "Sources:" and the source names you used."""


def build_context(chunks):
    """Join the retrieved chunks into one text block, labelled by source."""
    return "\n\n".join(f"[Source: {c['source']}]\n{c['text']}" for c in chunks)


def answer(question):
    """Retrieve relevant chunks and ask the LLM to answer from them."""
    chunks = retrieve(question)
    context = build_context(chunks)

    response = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return response.choices[0].message.content, chunks


if __name__ == "__main__":
    question = input("Ask a question: ")
    reply, chunks = answer(question)

    print("\nAnswer:\n" + reply)
    print("\nRetrieved from:", sorted({c["source"] for c in chunks}))