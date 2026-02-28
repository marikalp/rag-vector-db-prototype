from typing import Any

class Generator:
    """
    Wraps the interaction with the LLM.
    In this prototype, we only define the interface.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        """
        Given a prompt (query + context), returns a generated answer.
        """
        # In a real system, this would call an LLM API.
        raise NotImplementedError("LLM generation is not executed in this prototype.")

