from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path("data")

def load_pdfs():
    """Read every PDF in data/ and return a list of (text, source, page)."""
    pages = []
    for pdf_path in DATA_DIR.glob("*.pdf"):
        reader = PdfReader(pdf_path)
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"text": text, "source": pdf_path.name, "page": page_number})
    return pages

def chunk_pages(pages, chunk_size=500, overlap=50):
    """Split each page into small overlapping chunks, keeping source and page."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = []
    for p in pages:
        for piece in splitter.split_text(p["text"]):
            chunks.append({"text": piece, "source": p["source"], "page": p["page"]})
    return chunks

if __name__ == "__main__":
    pages = load_pdfs()
    chunks = chunk_pages(pages)
    print(f"Loaded {len(pages)} pages and made {len(chunks)} chunks")
    for c in chunks[:5]:
        print("-" * 60)
        print(f"{c['source']} (page {c['page']})")
        print(c["text"])