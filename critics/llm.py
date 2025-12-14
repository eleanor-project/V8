from abc import ABC, abstractmethod
from typing import Any, Dict


class LLMClient(ABC):
    @abstractmethod
    def invoke(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        Must return a JSON-serializable dict that can be parsed
        into the CriticOutput fields (except `critic`, which is injected
        by the caller).
        """
        raise NotImplementedError
