import time
from google.genai import errors
import sys
from dotenv import load_dotenv
from google import genai
from retrieve import search

load_dotenv()
client = genai.Client()   # reads GEMINI_API_KEY from your .env file
MODEL = "gemini-3.8-flash"   # if you get "model not found", see the note below


def ask(question, k=5):
    chunks = search(question, k=k)
    context = "\n\n".join(
        f"[{meta['source']}, page {meta['page']}]\n{text}" for text, meta in chunks
    )
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say \"I don't know based on the documents.\" "
        "Keep the answer short and clear.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    for attempt in range(5):
        try:
            response = client.models.generate_content(model=MODEL, contents=prompt)
            return response.text, chunks
        except errors.ServerError:
            wait = 10 * (attempt + 1)
            print(f"Model busy, retrying in {wait} seconds...")
            time.sleep(wait)
    return "The model is busy right now. Please try again in a few minutes.", chunks

if __name__ == "__main__":
    question = " ".join(sys.argv[1:])
    answer, chunks = ask(question)
    print("\nANSWER:\n" + answer)
    print("\nSOURCES:")
    for _, meta in chunks:
        print(f"- {meta['source']} (page {meta['page']})")