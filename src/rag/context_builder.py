from typing import List, Dict, Any

class ContextBuilder:
    """
    Builds a textual context from retrieved documents to be passed to the LLM.
    """

    def __init__(self, max_tokens: int = 1024):
        self.max_tokens = max_tokens

    def build_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Concatenate document contents into a single context string.
        Here we assume each document has a 'text' field.
        """
        texts = [doc.get("text", "") for doc in documents]
        context = "\n\n".join(texts)
        # In a real system, we would enforce max_tokens based on a tokenizer.
        return context

