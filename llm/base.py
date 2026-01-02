from abc import ABC, abstractmethod
from typing import Any, Dict


class LLMClient(ABC):
    """
    Abstract base class for all LLM backends used by ELEANOR.

    Implementations MUST:
    - Accept a system prompt and a user prompt
    - Return a dictionary compatible with critic schemas
    - Perform no aggregation or policy decisions
    """

    @abstractmethod
    def invoke(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        Execute a single reasoning call.

        Args:
            system_prompt: Role / charter / constraints
            user_prompt: The human input being evaluated

        Returns:
            A dict containing structured reasoning output
            (validated later by critic schemas)
        """
        raise NotImplementedError
