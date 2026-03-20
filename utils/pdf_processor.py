"""
PDF text extraction using PyPDF2.
Falls back gracefully if the library is not installed.
"""

import io


def extract_text_from_pdf(uploaded_file) -> str:
    """
    Extract raw text from a PDF uploaded via Streamlit's file_uploader.

    Parameters
    ----------
    uploaded_file : streamlit.runtime.uploaded_file_manager.UploadedFile

    Returns
    -------
    str  – concatenated text from all pages.
    """
    try:
        import PyPDF2
    except ImportError:
        raise ImportError(
            "PyPDF2 is required for PDF parsing. "
            "Install it with: pip install PyPDF2"
        )

    bytes_data = uploaded_file.read()
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(bytes_data))

    pages_text = []
    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text.strip())

    full_text = "\n\n".join(pages_text)

    if not full_text.strip():
        raise ValueError(
            "No text could be extracted from the PDF. "
            "The file may be a scanned image — try an OCR tool first."
        )

    return full_text
