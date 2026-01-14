"""
Audit ledger reader for querying historical audit events.

Provides read-only access to the immutable audit ledger with filtering,
pagination, and export capabilities.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    """Audit event type enumeration."""
    ACCESS_LOG = "access_log"
    SECRET_ACCESS = "secret_access_log"  # pragma: allowlist secret
    GOVERNANCE_DECISION = "governance_decision"
    CRITIC_EXECUTION = "critic_execution"
    VALIDATION_FAILURE = "validation_failure"
    SYSTEM_EVENT = "system_event"
    ALL = "all"


class AuditSeverity(str, Enum):
    """Audit event severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditLedgerReader:
    """
    Read-only audit ledger reader.
    
    Provides query capabilities for audit logs with filtering,
    pagination, and export functionality.
    """

    def __init__(self, ledger_path: Optional[Path] = None):
        """
        Initialize audit ledger reader.
        
        Args:
            ledger_path: Path to audit ledger file. Defaults to logs/audit.jsonl
        """
        self.ledger_path = ledger_path or Path("logs/audit.jsonl")
        self._ensure_ledger_exists()

    def _ensure_ledger_exists(self) -> None:
        """Ensure ledger file exists."""
        try:
            if not self.ledger_path.exists():
                self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
                self.ledger_path.touch()
        except Exception as e:
            logger.warning(f"Failed to initialize ledger: {e}")

    def query(
        self,
        *,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[str]] = None,
        severity: Optional[str] = None,
        user: Optional[str] = None,
        request_id: Optional[str] = None,
        resource: Optional[str] = None,
        search_text: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        sort_desc: bool = True,
    ) -> Dict[str, Any]:
        """
        Query audit ledger with filters.
        
        Args:
            start_time: Filter events after this time
            end_time: Filter events before this time
            event_types: Filter by event types
            severity: Filter by severity level
            user: Filter by user/accessor
            request_id: Filter by request ID
            resource: Filter by resource name
            search_text: Full-text search in event details
            limit: Maximum number of results
            offset: Results offset for pagination
            sort_desc: Sort descending (newest first)
            
        Returns:
            Dict with 'events', 'total', 'limit', 'offset' keys
        """
        try:
            events = self._read_events()
            
            # Apply filters
            filtered = self._apply_filters(
                events=events,
                start_time=start_time,
                end_time=end_time,
                event_types=event_types,
                severity=severity,
                user=user,
                request_id=request_id,
                resource=resource,
                search_text=search_text,
            )
            
            # Sort
            filtered.sort(
                key=lambda e: e.get("timestamp", ""),
                reverse=sort_desc
            )
            
            # Paginate
            total = len(filtered)
            paginated = filtered[offset:offset + limit]
            
            return {
                "events": paginated,
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total,
            }
        except Exception as e:
            logger.error(f"Failed to query audit ledger: {e}")
            return {
                "events": [],
                "total": 0,
                "limit": limit,
                "offset": offset,
                "error": str(e),
            }

    def _read_events(self) -> List[Dict[str, Any]]:
        """Read all events from ledger."""
        events = []
        try:
            if not self.ledger_path.exists():
                return events
                
            with open(self.ledger_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                        events.append(event)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON in ledger: {line[:100]}")
        except Exception as e:
            logger.error(f"Failed to read ledger: {e}")
        
        return events

    def _apply_filters(
        self,
        events: List[Dict[str, Any]],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[str]] = None,
        severity: Optional[str] = None,
        user: Optional[str] = None,
        request_id: Optional[str] = None,
        resource: Optional[str] = None,
        search_text: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Apply filters to events."""
        filtered = events
        
        # Time range filter
        if start_time:
            start_str = start_time.isoformat()
            filtered = [e for e in filtered if e.get("timestamp", "") >= start_str]
        
        if end_time:
            end_str = end_time.isoformat()
            filtered = [e for e in filtered if e.get("timestamp", "") <= end_str]
        
        # Event type filter
        if event_types and "all" not in [t.lower() for t in event_types]:
            filtered = [e for e in filtered if e.get("event") in event_types]
        
        # Severity filter
        if severity:
            filtered = [
                e for e in filtered
                if e.get("severity", "info").lower() == severity.lower()
            ]
        
        # User filter
        if user:
            filtered = [
                e for e in filtered
                if (
                    e.get("user") == user
                    or e.get("details", {}).get("user") == user
                    or e.get("details", {}).get("accessor") == user
                )
            ]
        
        # Request ID filter
        if request_id:
            filtered = [
                e for e in filtered
                if (
                    e.get("request_id") == request_id
                    or e.get("details", {}).get("request_id") == request_id
                )
            ]
        
        # Resource filter
        if resource:
            filtered = [
                e for e in filtered
                if (
                    e.get("resource") == resource
                    or e.get("details", {}).get("resource") == resource
                )
            ]
        
        # Text search filter
        if search_text:
            search_lower = search_text.lower()
            filtered = [
                e for e in filtered
                if search_lower in json.dumps(e).lower()
            ]
        
        return filtered

    def get_event_types(self) -> List[str]:
        """Get list of unique event types in ledger."""
        try:
            events = self._read_events()
            event_types = set(e.get("event") for e in events if e.get("event"))
            return sorted(event_types)
        except Exception as e:
            logger.error(f"Failed to get event types: {e}")
            return []

    def get_stats(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get audit statistics for recent period.
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Statistics dict with counts by event type and severity
        """
        try:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            result = self.query(start_time=cutoff, limit=999999)
            events = result["events"]
            
            # Count by event type
            by_type = {}
            for event in events:
                event_type = event.get("event", "unknown")
                by_type[event_type] = by_type.get(event_type, 0) + 1
            
            # Count by severity
            by_severity = {}
            for event in events:
                severity = event.get("severity", "info")
                by_severity[severity] = by_severity.get(severity, 0) + 1
            
            return {
                "period_hours": hours,
                "total_events": len(events),
                "by_type": by_type,
                "by_severity": by_severity,
                "start_time": cutoff.isoformat(),
                "end_time": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}

    def export(
        self,
        format: str = "json",
        **query_params
    ) -> str:
        """
        Export audit events to specified format.
        
        Args:
            format: Export format ('json' or 'csv')
            **query_params: Query parameters to filter events
            
        Returns:
            Exported data as string
        """
        result = self.query(**query_params)
        events = result["events"]
        
        if format == "json":
            return json.dumps(events, indent=2)
        elif format == "csv":
            return self._export_csv(events)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _export_csv(self, events: List[Dict[str, Any]]) -> str:
        """Export events to CSV format."""
        import csv
        import io
        
        if not events:
            return "timestamp,event,severity,details\n"
        
        output = io.StringIO()
        fieldnames = ["timestamp", "event", "severity", "user", "resource", "details"]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        
        writer.writeheader()
        for event in events:
            row = {
                "timestamp": event.get("timestamp", ""),
                "event": event.get("event", ""),
                "severity": event.get("severity", "info"),
                "user": event.get("details", {}).get("user", ""),
                "resource": event.get("details", {}).get("resource", ""),
                "details": json.dumps(event.get("details", {})),
            }
            writer.writerow(row)
        
        return output.getvalue()


# Global singleton instance
_ledger_reader = None


def get_ledger_reader() -> AuditLedgerReader:
    """Get global audit ledger reader instance."""
    global _ledger_reader
    if _ledger_reader is None:
        _ledger_reader = AuditLedgerReader()
    return _ledger_reader
