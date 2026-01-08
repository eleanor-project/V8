from abc import ABC, abstractmethod
from typing import Any, List, TYPE_CHECKING

if TYPE_CHECKING:
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
    async def write(self, record: "EvidenceRecord") -> Any:
        """Write a single evidence record."""
        ...

    async def write_batch(self, records: List["EvidenceRecord"]) -> Any:
        """
        Write multiple evidence records in batch.
        
        Default implementation calls write() for each record.
        Subclasses should override for optimized batch operations.
        
        Args:
            records: List of evidence records to write
            
        Returns:
            Result of batch write operation
        """
        # Default: write individually (subclasses should override)
        results = []
        for record in records:
            try:
                result = await self.write(record)
                results.append(result)
            except Exception as e:
                # Log error but continue with other records
                import logging
                logger = logging.getLogger(__name__)
                logger.error(
                    "batch_write_record_failed",
                    extra={"error": str(e), "trace_id": record.trace_id},
                )
                results.append(None)
        return results
