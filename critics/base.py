from abc import ABC, abstractmethod
from llm.base import LLMClient
from critics.schema import CriticOutput


class BaseCritic(ABC):
    name: str

    def __init__(self, llm: LLMClient):
        self.llm = llm

    @abstractmethod
    def system_prompt(self) -> str:
        pass

    def evaluate(self, prompt: str) -> CriticOutput:
        data = self.llm.invoke(
            system_prompt=self.system_prompt(),
            user_prompt=prompt
        )

        return CriticOutput(
            critic=self.name,
            **data
        )
