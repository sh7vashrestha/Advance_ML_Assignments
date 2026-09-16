import re
from pathlib import Path

from pypdf import PdfReader


REFERENCE_PATTERN = re.compile(
    r"(?im)^[ \t]*(references|bibliography)[ \t]*:?[ \t]*$"
)
APPENDIX_PATTERN = re.compile(
    r"(?im)^[ \t]*appendix(?:es)?(?:[ \t]+.*)?$"
)


def clean_text(text):
    """Remove repeated whitespace from extracted PDF text."""
    return re.sub(r"\s+", " ", text).strip()


def split_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping character chunks."""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def process_documents(pdf_folder, chunk_size=1000, overlap=200):
    """Extract and chunk every PDF in a directory."""
    pdf_folder = Path(pdf_folder)
    documents = []

    for pdf_path in sorted(pdf_folder.rglob("*.pdf")):
        reader = PdfReader(pdf_path)
        skipping_references = False

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            # References may appear before an appendix. Skip reference text,
            # but resume processing if a later appendix heading is found.
            if skipping_references:
                appendix_match = APPENDIX_PATTERN.search(text)
                if not appendix_match:
                    continue

                text = text[appendix_match.start():]
                skipping_references = False

            reference_match = REFERENCE_PATTERN.search(text)
            if reference_match:
                appendix_match = APPENDIX_PATTERN.search(
                    text,
                    reference_match.end(),
                )

                if appendix_match:
                    text = (
                        text[:reference_match.start()]
                        + "\n"
                        + text[appendix_match.start():]
                    )
                else:
                    text = text[:reference_match.start()]
                    skipping_references = True

            text = clean_text(text)

            page_chunks = split_text(
                text,
                chunk_size=chunk_size,
                overlap=overlap,
            )
            print(f"Processed {pdf_path.name}, page {page_number}: {len(page_chunks)} chunks")

            for chunk_number, chunk_text in enumerate(page_chunks):
                documents.append({
                    "text": chunk_text,
                    "source": pdf_path.name,
                    "page": page_number,
                    "chunk": chunk_number,
                })

    return documents
