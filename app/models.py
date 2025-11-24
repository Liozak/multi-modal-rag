"""
Pydantic models (data structures) used in the project.
"""

from typing import Optional, Literal
from pydantic import BaseModel

# Allowed modalities for chunks
Modality = Literal["text", "table", "image_ocr", "image"]


class DocumentChunk(BaseModel):
    """
    Represents a small piece ('chunk') of a document.

    - id: unique ID for this chunk
    - doc_id: which document it belongs to (e.g., 'sample_doc')
    - modality: type of content ('text', 'table', 'image_ocr', 'image')
    - page: page number in the original PDF (1-based)
    - content: the actual text content of this chunk
    - extra: optional dictionary for additional metadata (e.g., image path)
    """
    id: str
    doc_id: str
    modality: Modality
    page: int
    content: str
    extra: Optional[dict] = None
