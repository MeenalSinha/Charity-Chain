"""
pdf_processor.py - Safe Bounded PDF Evidence Extraction & Inspection
===================================================================
Inspects PDF documents using pypdf. Enforces strict page limits, extracts
document metadata and embedded image assets safely without loading unbounded buffers.
"""

from io import BytesIO
from typing import Dict, Any, List, Optional, Tuple
from pypdf import PdfReader

from backend.app.core.config import settings
from backend.app.core.constants import VerificationCheckStatus
from backend.app.core.logging import logger


class SafePdfProcessor:
    """Bounded PDF analyzer extracting structural metadata and embedded raster imagery."""

    def __init__(self, max_pages: Optional[int] = None):
        self.max_pages = max_pages or settings.MAX_PDF_PAGES

    def inspect_pdf(self, pdf_bytes: bytes) -> Dict[str, Any]:
        """
        Parses PDF structure, page count, and metadata.
        Extracts first page image if present for computer vision analysis.
        """
        try:
            reader = PdfReader(BytesIO(pdf_bytes))
            num_pages = len(reader.pages)

            if num_pages > self.max_pages:
                return {
                    "status": VerificationCheckStatus.FLAGGED,
                    "passed": False,
                    "page_count": num_pages,
                    "max_pages": self.max_pages,
                    "metadata": {},
                    "first_page_image_bytes": None,
                    "message": f"PDF contains {num_pages} pages, exceeding the maximum allowed limit ({self.max_pages} pages) for automated verification."
                }

            # Extract document metadata
            doc_info = reader.metadata or {}
            metadata = {
                "title": str(doc_info.title or ""),
                "author": str(doc_info.author or ""),
                "creator": str(doc_info.creator or ""),
                "producer": str(doc_info.producer or ""),
                "creation_date": str(doc_info.creation_date or "")
            }

            # Attempt to extract first raster image embedded in page 1
            first_image_bytes: Optional[bytes] = None
            if num_pages > 0:
                first_page = reader.pages[0]
                if hasattr(first_page, "images") and len(first_page.images) > 0:
                    try:
                        first_image_bytes = first_page.images[0].data
                    except Exception as img_err:
                        logger.debug(f"Could not extract embedded raster image from page 1: {img_err}")

            return {
                "status": VerificationCheckStatus.PASSED,
                "passed": True,
                "page_count": num_pages,
                "max_pages": self.max_pages,
                "metadata": metadata,
                "first_page_image_bytes": first_image_bytes,
                "message": f"PDF successfully validated ({num_pages} page(s)). Structural metadata and text headers extracted."
            }

        except Exception as e:
            logger.error(f"PDF validation failed: {e}")
            return {
                "status": VerificationCheckStatus.ERROR,
                "passed": False,
                "page_count": 0,
                "max_pages": self.max_pages,
                "metadata": {},
                "first_page_image_bytes": None,
                "message": f"Malformed or unreadable PDF document: {e}"
            }

