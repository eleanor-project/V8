from abc import ABC, abstractmethod
from typing import Any
from .evidence_recorder import EvidenceRecord

class EvidenceDBSink(ABC):
    """
    Abstract DB sink interface. Implementations may write to:
        - PostgreSQL
        - MongoDB
        - ElasticSearch
        - DynamoDB
        - Analytics pipelines
    """

    @abstractmethod
    async def write(self, record: EvidenceRecord) -> Any:
        ...
