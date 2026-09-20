from io import BytesIO

from pypdf import PdfReader

from app.ingestion.models import ExtractedPage


class IngestionValidationError(ValueError):
    """Raised for an upload that cannot be safely ingested."""


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def extract_pages(filename: str, content: bytes) -> list[ExtractedPage]:
    suffix = filename.lower().rsplit(".", maxsplit=1)
    extension = f".{suffix[-1]}" if len(suffix) == 2 else ""
    if extension not in SUPPORTED_EXTENSIONS:
        raise IngestionValidationError("Only PDF, TXT, and Markdown files are currently supported.")

    if extension == ".pdf":
        return _extract_pdf(content)
    return _extract_text(content)


def _extract_pdf(content: bytes) -> list[ExtractedPage]:
    try:
        reader = PdfReader(BytesIO(content))
        if reader.is_encrypted:
            raise IngestionValidationError("Password-protected PDFs are not supported.")
        pages = [
            ExtractedPage(page_number=index, text=(page.extract_text() or "").strip())
            for index, page in enumerate(reader.pages, start=1)
        ]
    except IngestionValidationError:
        raise
    except Exception as exc:
        raise IngestionValidationError("The PDF could not be read.") from exc

    pages = [page for page in pages if page.text]
    if not pages:
        raise IngestionValidationError("No extractable text was found in this PDF.")
    return pages


def _extract_text(content: bytes) -> list[ExtractedPage]:
    try:
        text = content.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise IngestionValidationError("Text files must be UTF-8 encoded.") from exc
    if not text:
        raise IngestionValidationError("The uploaded file is empty.")
    return [ExtractedPage(page_number=1, text=text)]
