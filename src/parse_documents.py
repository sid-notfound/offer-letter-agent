import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict
import re


def clean_text(text: str) -> str:
    """Clean extracted text."""
    text = re.sub(r"\s{2,}", " ", text)  # Remove excessive spaces
    return text.strip()


def chunk_policy_text(text: str, filename: str) -> List[Dict]:
    """Split text into logical chunks using section headings as anchors."""
    # Use headings or emoji bullets as split markers
    chunks = []
    lines = text.split("\n")
    current_chunk = ""
    current_title = "General"
    
    for line in lines:
        line = line.strip()
        if re.match(r"^(📘|🏷|🧠|🧾|🏢|🏡|🧍|✅|❌|🚨|🔄|📌|🛫|💰|📅|🏨|🌐|🛡|📋)", line):
            if current_chunk:
                chunks.append({
                    "content": clean_text(current_chunk),
                    "section": current_title,
                    "source": filename
                })
            current_title = line
            current_chunk = ""
        current_chunk += line + "\n"
    
    if current_chunk:
        chunks.append({
            "content": clean_text(current_chunk),
            "section": current_title,
            "source": filename
        })

    return chunks


def parse_pdf(file_path: str) -> List[Dict]:
    """Extract text from a PDF and chunk it."""
    doc = fitz.open(file_path)
    all_chunks = []

    for page_num, page in enumerate(doc):
        text = page.get_text()
        filename = Path(file_path).name
        page_chunks = chunk_policy_text(text, filename)
        for chunk in page_chunks:
            chunk["page"] = page_num + 1
        all_chunks.extend(page_chunks)

    return all_chunks


def parse_all_documents(data_folder: str = "data") -> List[Dict]:
    """Process all relevant PDFs into chunks."""
    files = [
        "hr_leave_policy.pdf",
        "hr_travel_policy.pdf",
        "offer_letter_sample.pdf"
    ]
    
    all_chunks = []
    for fname in files:
        file_path = str(Path(data_folder) / fname)
        chunks = parse_pdf(file_path)
        print(f"Parsed {fname}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    return all_chunks


if __name__ == "__main__":
    chunks = parse_all_documents()
    print(f"Total chunks: {len(chunks)}")
    print("Sample:", chunks[0])
