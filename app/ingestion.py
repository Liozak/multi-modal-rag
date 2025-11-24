"""
Functions to read PDFs and convert them into DocumentChunk objects.

- STEP 3: extract_text_from_pdf (for debugging)
- STEP 3: get_pdf_page_texts (for chunking)
- STEP 8: extract_image_ocr_chunks (for images + OCR)
"""

from pathlib import Path
from typing import List
import io

import fitz  # PyMuPDF
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


import tabula


from app.config import RAW_DOCS_DIR
from app.models import DocumentChunk


def extract_text_from_pdf(pdf_name: str = "sample.pdf", max_pages: int = 3):
    """
    Reads a PDF from data/raw_docs and prints text from each page.
    max_pages = how many pages to show (for testing)
    """
    pdf_path = Path(RAW_DOCS_DIR) / pdf_name

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    print(f"\n✅ Opening PDF: {pdf_path}\n")

    document = fitz.open(pdf_path)

    total_pages = len(document)
    print(f"Total Pages in PDF: {total_pages}\n")

    for page_number in range(min(max_pages, total_pages)):
        page = document[page_number]
        text = page.get_text()

        print("=" * 50)
        print(f"📄 PAGE {page_number + 1}")
        print("=" * 50)
        print(text[:1500])
        print("\n\n")


def get_pdf_page_texts(pdf_name: str = "sample.pdf") -> List[str]:
    """
    Returns a list of strings, one per page of the PDF.
    We use this for chunking and building the QA index.
    """
    pdf_path = Path(RAW_DOCS_DIR) / pdf_name

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    document = fitz.open(pdf_path)
    page_texts: List[str] = []

    for page_number in range(len(document)):
        page = document[page_number]
        text = page.get_text()
        page_texts.append(text)

    return page_texts


def extract_image_ocr_chunks(
    pdf_name: str = "sample.pdf",
    doc_id: str = "sample_doc",
) -> List[DocumentChunk]:
    """
    Extracts images from the PDF and runs OCR on them.
    Returns a list of DocumentChunk objects with modality='image_ocr'.
    """
    pdf_path = Path(RAW_DOCS_DIR) / pdf_name

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    document = fitz.open(pdf_path)
    chunks: List[DocumentChunk] = []

    for page_index in range(len(document)):
        page = document[page_index]
        image_list = page.get_images(full=True)

        if not image_list:
            continue  # no images on this page

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = document.extract_image(xref)
            image_bytes = base_image["image"]

            # Load image with PIL
            img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # Run OCR
            ocr_text = pytesseract.image_to_string(img_pil)

            if not ocr_text.strip():
                ocr_text = "[NO TEXT DETECTED IN IMAGE]"

            chunk = DocumentChunk(
                id=f"img_ocr_{page_index+1}_{img_index}",
                doc_id=doc_id,
                modality="image_ocr",
                page=page_index + 1,
                content=ocr_text,
                extra={"image_index": img_index},
            )
            chunks.append(chunk)

    return chunks

def extract_table_chunks(
    pdf_name: str = "sample.pdf",
    doc_id: str = "sample_doc",
) -> List[DocumentChunk]:
    """
    Extracts tables from the PDF using tabula-py and converts them
    into DocumentChunk objects with modality='table'.
    """
    pdf_path = Path(RAW_DOCS_DIR) / pdf_name

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    # tabula returns a list of DataFrames
    try:
        dfs = tabula.read_pdf(str(pdf_path), pages="all", multiple_tables=True)
    except Exception as e:
        print(f"[WARN] Table extraction failed: {e}")
        return []

    chunks: List[DocumentChunk] = []

    for idx, df in enumerate(dfs):
        # Convert DataFrame to a readable text representation
        table_text = df.to_csv(index=False)

        chunk = DocumentChunk(
            id=f"table_{idx}",
            doc_id=doc_id,
            modality="table",
            page=-1,  # tabula doesn't always give page; you can refine later
            content=table_text,
            extra={"n_rows": df.shape[0], "n_cols": df.shape[1]},
        )
        chunks.append(chunk)

    return chunks



if __name__ == "__main__":
    # simple manual test
    extract_text_from_pdf()
