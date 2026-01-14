"""
Audit query interface for investigations and compliance reporting.

Provides powerful query capabilities over audit logs with:
- Time-range filtering
- Multi-field filtering
- Full-text search
- Aggregations and analytics
- Compliance report generation
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from engine.security.audit.core import AuditLevel, ComplianceFramework, EventCategory
from engine.security.ledger import LedgerRecord, get_ledger_reader


class QueryOperator(str, Enum):
    """Query operators for filtering."""

    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "lte"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    REGEX = "regex"


@dataclass
class QueryFilter:
    """Single filter condition."""

    field: str
    operator: QueryOperator
    value: Any

    def matches(self, record: Dict[str, Any]) -> bool:
        """Check if record matches this filter."""
        field_value = self._get_nested_value(record, self.field)
        if field_value is None:
            return False

        if self.operator == QueryOperator.EQUALS:
            return field_value == self.value
        elif self.operator == QueryOperator.NOT_EQUALS:
            return field_value != self.value
        elif self.operator == QueryOperator.GREATER_THAN:
            return field_value > self.value
        elif self.operator == QueryOperator.GREATER_THAN_OR_EQUAL:
            return field_value >= self.value
        elif self.operator == QueryOperator.LESS_THAN:
            return field_value < self.value
        elif self.operator == QueryOperator.LESS_THAN_OR_EQUAL:
            return field_value <= self.value
        elif self.operator == QueryOperator.IN:
            return field_value in self.value
        elif self.operator == QueryOperator.NOT_IN:
            return field_value not in self.value
        elif self.operator == QueryOperator.CONTAINS:
            return str(self.value) in str(field_value)
        elif self.operator == QueryOperator.STARTS_WITH:
            return str(field_value).startswith(str(self.value))
        elif self.operator == QueryOperator.ENDS_WITH:
            return str(field_value).endswith(str(self.value))
        elif self.operator == QueryOperator.REGEX:
            import re
            return bool(re.search(self.value, str(field_value)))
        return False

    def _get_nested_value(self, record: Dict[str, Any], field_path: str) -> Any:
        """Get nested field value using dot notation."""
        parts = field_path.split(".")
        value = record
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None
            if value is None:
                return None
        return value


@dataclass
class AuditQuery:
    """Audit log query with filters and options."""

    # Time range
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Filters
    filters: List[QueryFilter] = field(default_factory=list)

    # Event type and category
    event_types: List[str] = field(default_factory=list)
    categories: List[EventCategory] = field(default_factory=list)
    levels: List[AuditLevel] = field(default_factory=list)

    # Actor and resource
    actor_ids: List[str] = field(default_factory=list)
    resource_types: List[str] = field(default_factory=list)
    resource_ids: List[str] = field(default_factory=list)

    # Compliance
    compliance_frameworks: List[ComplianceFramework] = field(default_factory=list)

    # Multi-tenancy
    tenant_ids: List[str] = field(default_factory=list)
    organization_ids: List[str] = field(default_factory=list)

    # Full-text search
    search_text: Optional[str] = None

    # Pagination
    limit: Optional[int] = None
    offset: int = 0

    # Sorting
    sort_by: str = "timestamp"
    sort_desc: bool = True

    def matches(self, record: Dict[str, Any]) -> bool:
        """Check if record matches all query conditions."""
        payload = record.get("payload", {})

        # Time range check
        if self.start_time or self.end_time:
            timestamp_str = payload.get("timestamp")
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(
                        timestamp_str.replace("Z", "+00:00")
                    )
                    if self.start_time and timestamp < self.start_time:
                        return False
                    if self.end_time and timestamp > self.end_time:
                        return False
                except ValueError:
                    pass

        # Event type check
        if self.event_types:
            if record.get("event") not in self.event_types:
                return False

        # Category check
        if self.categories:
            category = payload.get("category")
            if category not in [c.value for c in self.categories]:
                return False

        # Level check
        if self.levels:
            level = payload.get("level")
            if level not in [l.value for l in self.levels]:
                return False

        # Actor check
        if self.actor_ids:
            actor_id = payload.get("actor_id") or record.get("actor_id")
            if actor_id not in self.actor_ids:
                return False

        # Resource checks
        if self.resource_types:
            if payload.get("resource_type") not in self.resource_types:
                return False

        if self.resource_ids:
            if payload.get("resource_id") not in self.resource_ids:
                return False

        # Compliance framework check
        if self.compliance_frameworks:
            frameworks = payload.get("compliance_frameworks", [])
            if not any(f in frameworks for f in [cf.value for cf in self.compliance_frameworks]):
                return False

        # Tenant/org check
        if self.tenant_ids:
            if payload.get("tenant_id") not in self.tenant_ids:
                return False

        if self.organization_ids:
            if payload.get("organization_id") not in self.organization_ids:
                return False

        # Custom filters
        for filter_condition in self.filters:
            if not filter_condition.matches(payload):
                return False

        # Full-text search
        if self.search_text:
            import json
            record_text = json.dumps(payload).lower()
            if self.search_text.lower() not in record_text:
                return False

        return True


class AuditQueryBuilder:
    """Fluent builder for audit queries."""

    def __init__(self):
        self.query = AuditQuery()

    def time_range(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> "AuditQueryBuilder":
        """Set time range filter."""
        self.query.start_time = start
        self.query.end_time = end
        return self

    def last_hours(self, hours: int) -> "AuditQueryBuilder":
        """Filter events from last N hours."""
        now = datetime.now(timezone.utc)
        self.query.end_time = now
        self.query.start_time = now - timedelta(hours=hours)
        return self

    def last_days(self, days: int) -> "AuditQueryBuilder":
        """Filter events from last N days."""
        now = datetime.now(timezone.utc)
        self.query.end_time = now
        self.query.start_time = now - timedelta(days=days)
        return self

    def event_types(self, *types: str) -> "AuditQueryBuilder":
        """Filter by event types."""
        self.query.event_types = list(types)
        return self

    def categories(self, *categories: EventCategory) -> "AuditQueryBuilder":
        """Filter by categories."""
        self.query.categories = list(categories)
        return self

    def levels(self, *levels: AuditLevel) -> "AuditQueryBuilder":
        """Filter by severity levels."""
        self.query.levels = list(levels)
        return self

    def critical_only(self) -> "AuditQueryBuilder":
        """Filter critical events only."""
        return self.levels(AuditLevel.CRITICAL)

    def errors_and_above(self) -> "AuditQueryBuilder":
        """Filter errors and critical events."""
        return self.levels(AuditLevel.ERROR, AuditLevel.CRITICAL)

    def actors(self, *actor_ids: str) -> "AuditQueryBuilder":
        """Filter by actor IDs."""
        self.query.actor_ids = list(actor_ids)
        return self

    def resources(
        self,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
    ) -> "AuditQueryBuilder":
        """Filter by resource."""
        if resource_type:
            self.query.resource_types = [resource_type]
        if resource_id:
            self.query.resource_ids = [resource_id]
        return self

    def compliance(self, *frameworks: ComplianceFramework) -> "AuditQueryBuilder":
        """Filter by compliance frameworks."""
        self.query.compliance_frameworks = list(frameworks)
        return self

    def tenant(self, tenant_id: str) -> "AuditQueryBuilder":
        """Filter by tenant."""
        self.query.tenant_ids = [tenant_id]
        return self

    def search(self, text: str) -> "AuditQueryBuilder":
        """Full-text search."""
        self.query.search_text = text
        return self

    def filter(
        self,
        field: str,
        operator: QueryOperator,
        value: Any,
    ) -> "AuditQueryBuilder":
        """Add custom filter."""
        self.query.filters.append(QueryFilter(field, operator, value))
        return self

    def limit(self, count: int) -> "AuditQueryBuilder":
        """Limit results."""
        self.query.limit = count
        return self

    def offset(self, count: int) -> "AuditQueryBuilder":
        """Set result offset."""
        self.query.offset = count
        return self

    def sort_by(self, field: str, descending: bool = True) -> "AuditQueryBuilder":
        """Set sorting."""
        self.query.sort_by = field
        self.query.sort_desc = descending
        return self

    def build(self) -> AuditQuery:
        """Build and return the query."""
        return self.query

    def execute(self) -> List[Dict[str, Any]]:
        """Build and execute the query."""
        return execute_query(self.query)


def execute_query(query: AuditQuery) -> List[Dict[str, Any]]:
    """Execute an audit query against the ledger."""
    reader = get_ledger_reader()
    records = reader.read_all()

    # Convert LedgerRecord to dict and filter
    results = []
    for record in records:
        record_dict = {
            "event_id": record.event_id,
            "timestamp": record.timestamp,
            "event": record.event,
            "trace_id": record.trace_id,
            "actor_id": record.actor_id,
            "payload": record.payload,
            "payload_hash": record.payload_hash,
            "prev_hash": record.prev_hash,
            "record_hash": record.record_hash,
        }

        if query.matches(record_dict):
            results.append(record_dict)

    # Sort results
    def get_sort_key(r: Dict[str, Any]) -> Any:
        if query.sort_by == "timestamp":
            return r.get("timestamp", "")
        return r.get("payload", {}).get(query.sort_by, "")

    results.sort(key=get_sort_key, reverse=query.sort_desc)

    # Apply pagination
    start = query.offset
    end = start + query.limit if query.limit else None
    return results[start:end]


def generate_compliance_report(
    framework: ComplianceFramework,
    start_date: datetime,
    end_date: datetime,
) -> Dict[str, Any]:
    """Generate compliance report for a specific framework."""
    query = (
        AuditQueryBuilder()
        .time_range(start_date, end_date)
        .compliance(framework)
        .build()
    )

    results = execute_query(query)

    # Aggregate statistics
    stats = {
        "framework": framework.value,
        "period_start": start_date.isoformat(),
        "period_end": end_date.isoformat(),
        "total_events": len(results),
        "by_level": {},
        "by_category": {},
        "by_event_type": {},
        "security_incidents": 0,
        "failed_authentications": 0,
        "unauthorized_access_attempts": 0,
        "data_access_events": 0,
        "configuration_changes": 0,
    }

    for result in results:
        payload = result.get("payload", {})

        # By level
        level = payload.get("level", "unknown")
        stats["by_level"][level] = stats["by_level"].get(level, 0) + 1

        # By category
        category = payload.get("category", "unknown")
        stats["by_category"][category] = stats["by_category"].get(category, 0) + 1

        # By event type
        event_type = result.get("event", "unknown")
        stats["by_event_type"][event_type] = (
            stats["by_event_type"].get(event_type, 0) + 1
        )

        # Specific metrics
        if category == EventCategory.SECURITY.value:
            stats["security_incidents"] += 1
        if event_type == "authentication" and payload.get("outcome") == "failure":
            stats["failed_authentications"] += 1
        if event_type == "authorization" and not payload.get("permission_granted", True):
            stats["unauthorized_access_attempts"] += 1
        if category == EventCategory.DATA_ACCESS.value:
            stats["data_access_events"] += 1
        if category == EventCategory.CONFIGURATION.value:
            stats["configuration_changes"] += 1

    return stats


__all__ = [
    "AuditQuery",
    "AuditQueryBuilder",
    "QueryFilter",
    "QueryOperator",
    "execute_query",
    "generate_compliance_report",
]
