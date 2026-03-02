class ContextBuilder:
    def __init__(self, max_tokens: int = 1024):
        self.max_tokens = max_tokens

    def build_context(self, documents):
        texts = [doc.get("text", "") for doc in documents]
        return "\n\n".join(texts)
