"""
SIEM (Security Information and Event Management) integration.

Provides exporters for major SIEM platforms:
- Splunk
- Elastic (ELK)
- Datadog
- Azure Sentinel
- AWS Security Hub
- Google Chronicle
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import requests

from engine.security.audit.core import AuditEvent

logger = logging.getLogger(__name__)


class SIEMExporter(ABC):
    """Base class for SIEM exporters."""

    def __init__(self, name: str):
        self.name = name
        self.enabled = False
        self._export_count = 0
        self._error_count = 0

    @abstractmethod
    def export(self, event: AuditEvent, sanitized_data: Dict[str, Any]) -> bool:
        """Export event to SIEM platform.

        Args:
            event: The audit event object
            sanitized_data: Sanitized event data dictionary

        Returns:
            True if export succeeded, False otherwise
        """
        pass

    def get_metrics(self) -> Dict[str, Any]:
        """Get exporter metrics."""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "exports": self._export_count,
            "errors": self._error_count,
            "success_rate": (
                (self._export_count - self._error_count) / self._export_count
                if self._export_count > 0
                else 1.0
            ),
        }


class SplunkExporter(SIEMExporter):
    """Splunk HEC (HTTP Event Collector) exporter."""

    def __init__(self):
        super().__init__("splunk")
        self.hec_url = os.getenv("SPLUNK_HEC_URL")
        self.hec_token = os.getenv("SPLUNK_HEC_TOKEN")
        self.index = os.getenv("SPLUNK_INDEX", "audit")
        self.sourcetype = os.getenv("SPLUNK_SOURCETYPE", "eleanor:audit")
        self.verify_ssl = os.getenv("SPLUNK_VERIFY_SSL", "true").lower() == "true"

        self.enabled = bool(self.hec_url and self.hec_token)

        if self.enabled:
            self.session = requests.Session()
            self.session.headers.update({
                "Authorization": f"Splunk {self.hec_token}",
                "Content-Type": "application/json",
            })

    def export(self, event: AuditEvent, sanitized_data: Dict[str, Any]) -> bool:
        """Export to Splunk via HEC."""
        if not self.enabled:
            return False

        try:
            payload = {
                "time": sanitized_data.get("timestamp"),
                "host": sanitized_data.get("hostname"),
                "source": sanitized_data.get("service_name"),
                "sourcetype": self.sourcetype,
                "index": self.index,
                "event": sanitized_data,
            }

            response = self.session.post(
                urljoin(self.hec_url, "/services/collector/event"),
                json=payload,
                timeout=5,
                verify=self.verify_ssl,
            )
            response.raise_for_status()

            self._export_count += 1
            return True

        except Exception as exc:
            self._error_count += 1
            logger.debug("splunk_export_failed", extra={"error": str(exc)})
            return False


class ElasticExporter(SIEMExporter):
    """Elasticsearch exporter for ELK stack."""

    def __init__(self):
        super().__init__("elastic")
        self.es_url = os.getenv("ELASTIC_URL")
        self.api_key = os.getenv("ELASTIC_API_KEY")
        self.username = os.getenv("ELASTIC_USERNAME")
        self.password = os.getenv("ELASTIC_PASSWORD")
        self.index_prefix = os.getenv("ELASTIC_INDEX_PREFIX", "eleanor-audit")
        self.verify_ssl = os.getenv("ELASTIC_VERIFY_SSL", "true").lower() == "true"

        self.enabled = bool(self.es_url and (self.api_key or (self.username and self.password)))

        if self.enabled:
            self.session = requests.Session()
            if self.api_key:
                self.session.headers.update({"Authorization": f"ApiKey {self.api_key}"})
            elif self.username and self.password:
                self.session.auth = (self.username, self.password)
            self.session.headers.update({"Content-Type": "application/json"})

    def export(self, event: AuditEvent, sanitized_data: Dict[str, Any]) -> bool:
        """Export to Elasticsearch."""
        if not self.enabled:
            return False

        try:
            # Use date-based index for time-series data
            from datetime import datetime

            date_str = datetime.now().strftime("%Y.%m.%d")
            index_name = f"{self.index_prefix}-{date_str}"

            response = self.session.post(
                urljoin(self.es_url, f"/{index_name}/_doc"),
                json=sanitized_data,
                timeout=5,
                verify=self.verify_ssl,
            )
            response.raise_for_status()

            self._export_count += 1
            return True

        except Exception as exc:
            self._error_count += 1
            logger.debug("elastic_export_failed", extra={"error": str(exc)})
            return False


class DatadogExporter(SIEMExporter):
    """Datadog Security Monitoring exporter."""

    def __init__(self):
        super().__init__("datadog")
        self.api_key = os.getenv("DATADOG_API_KEY")
        self.site = os.getenv("DATADOG_SITE", "datadoghq.com")
        self.service_name = os.getenv("DD_SERVICE", "eleanor-v8")

        self.enabled = bool(self.api_key)

        if self.enabled:
            self.session = requests.Session()
            self.session.headers.update({
                "DD-API-KEY": self.api_key,
                "Content-Type": "application/json",
            })
            self.logs_url = f"https://http-intake.logs.{self.site}/api/v2/logs"

    def export(self, event: AuditEvent, sanitized_data: Dict[str, Any]) -> bool:
        """Export to Datadog."""
        if not self.enabled:
            return False

        try:
            # Format for Datadog logs API
            payload = {
                "ddsource": "eleanor",
                "ddtags": f"env:{sanitized_data.get('environment')},service:{self.service_name}",
                "hostname": sanitized_data.get("hostname"),
                "message": json.dumps(sanitized_data),
                "status": self._map_level_to_status(event.level.value),
            }

            response = self.session.post(
                self.logs_url,
                json=[payload],
                timeout=5,
            )
            response.raise_for_status()

            self._export_count += 1
            return True

        except Exception as exc:
            self._error_count += 1
            logger.debug("datadog_export_failed", extra={"error": str(exc)})
            return False

    def _map_level_to_status(self, level: str) -> str:
        """Map audit level to Datadog status."""
        mapping = {
            "debug": "debug",
            "info": "info",
            "warning": "warn",
            "error": "error",
            "critical": "critical",
        }
        return mapping.get(level, "info")


class AzureSentinelExporter(SIEMExporter):
    """Azure Sentinel (Log Analytics) exporter."""

    def __init__(self):
        super().__init__("azure_sentinel")
        self.workspace_id = os.getenv("AZURE_SENTINEL_WORKSPACE_ID")
        self.shared_key = os.getenv("AZURE_SENTINEL_SHARED_KEY")
        self.log_type = os.getenv("AZURE_SENTINEL_LOG_TYPE", "EleanorAudit")

        self.enabled = bool(self.workspace_id and self.shared_key)

    def export(self, event: AuditEvent, sanitized_data: Dict[str, Any]) -> bool:
        """Export to Azure Sentinel."""
        if not self.enabled:
            return False

        try:
            import base64
            import hashlib
            import hmac
            from datetime import datetime

            # Build the API signature
            date_string = datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")
            body = json.dumps([sanitized_data])
            content_length = len(body)

            string_to_hash = (
                f"POST\n{content_length}\napplication/json\n"
                f"x-ms-date:{date_string}\n/api/logs"
            )
            bytes_to_hash = bytes(string_to_hash, "UTF-8")
            decoded_key = base64.b64decode(self.shared_key)
            encoded_hash = base64.b64encode(
                hmac.new(decoded_key, bytes_to_hash, digestmod=hashlib.sha256).digest()
            ).decode()

            authorization = f"SharedKey {self.workspace_id}:{encoded_hash}"

            headers = {
                "Content-Type": "application/json",
                "Log-Type": self.log_type,
                "x-ms-date": date_string,
                "Authorization": authorization,
            }

            url = (
                f"https://{self.workspace_id}.ods.opinsights.azure.com"
                f"/api/logs?api-version=2016-04-01"
            )

            response = requests.post(url, data=body, headers=headers, timeout=5)
            response.raise_for_status()

            self._export_count += 1
            return True

        except Exception as exc:
            self._error_count += 1
            logger.debug("azure_sentinel_export_failed", extra={"error": str(exc)})
            return False


class AWSSecurityHubExporter(SIEMExporter):
    """AWS Security Hub exporter."""

    def __init__(self):
        super().__init__("aws_security_hub")
        self.region = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION"))
        self.account_id = os.getenv("AWS_ACCOUNT_ID")

        # Check if boto3 is available
        try:
            import boto3
            self.boto3 = boto3
            self.enabled = bool(self.region and self.account_id)
            if self.enabled:
                self.client = boto3.client("securityhub", region_name=self.region)
        except ImportError:
            self.boto3 = None
            self.enabled = False

    def export(self, event: AuditEvent, sanitized_data: Dict[str, Any]) -> bool:
        """Export to AWS Security Hub."""
        if not self.enabled or not self.boto3:
            return False

        try:
            from datetime import datetime

            # Map to ASFF (AWS Security Finding Format)
            finding = {
                "SchemaVersion": "2018-10-08",
                "Id": f"eleanor-audit/{event.event_id}",
                "ProductArn": (
                    f"arn:aws:securityhub:{self.region}:{self.account_id}:"
                    f"product/{self.account_id}/default"
                ),
                "GeneratorId": "eleanor-audit",
                "AwsAccountId": self.account_id,
                "Types": ["Software and Configuration Checks/AWS Security Best Practices"],
                "CreatedAt": sanitized_data.get("timestamp"),
                "UpdatedAt": datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
                "Severity": {"Label": self._map_level_to_severity(event.level.value)},
                "Title": event.event_type,
                "Description": json.dumps(sanitized_data),
                "Resources": [
                    {
                        "Type": "Other",
                        "Id": event.resource_id or "unknown",
                    }
                ],
            }

            self.client.batch_import_findings(Findings=[finding])

            self._export_count += 1
            return True

        except Exception as exc:
            self._error_count += 1
            logger.debug("aws_security_hub_export_failed", extra={"error": str(exc)})
            return False

    def _map_level_to_severity(self, level: str) -> str:
        """Map audit level to AWS Security Hub severity."""
        mapping = {
            "debug": "INFORMATIONAL",
            "info": "LOW",
            "warning": "MEDIUM",
            "error": "HIGH",
            "critical": "CRITICAL",
        }
        return mapping.get(level, "MEDIUM")


def get_siem_exporters() -> List[SIEMExporter]:
    """Get list of configured SIEM exporters."""
    exporters: List[SIEMExporter] = []

    # Initialize all exporters
    exporter_classes = [
        SplunkExporter,
        ElasticExporter,
        DatadogExporter,
        AzureSentinelExporter,
        AWSSecurityHubExporter,
    ]

    for exporter_class in exporter_classes:
        try:
            exporter = exporter_class()
            if exporter.enabled:
                exporters.append(exporter)
                logger.info(
                    "siem_exporter_enabled",
                    extra={"siem": exporter.name},
                )
        except Exception as exc:
            logger.warning(
                "siem_exporter_init_failed",
                extra={
                    "siem": exporter_class.__name__,
                    "error": str(exc),
                },
            )

    return exporters


__all__ = [
    "SIEMExporter",
    "SplunkExporter",
    "ElasticExporter",
    "DatadogExporter",
    "AzureSentinelExporter",
    "AWSSecurityHubExporter",
    "get_siem_exporters",
]
