import re
from typing import List
from pypdf import PdfReader
import nltk

class PDFLoader:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_pdf(self, path: str) -> str:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return self.clean_text(text)

    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def chunk_text(self, text: str) -> List[str]:
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = []

        for sentence in sentences:
            if sum(len(s) for s in current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk.append(sentence)
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
