"""
Real-time alerting system for critical audit events.

Supports multiple alert channels:
- Email (SMTP)
- Slack
- PagerDuty
- Microsoft Teams
- Generic Webhooks
"""

import json
import logging
import os
import smtplib
from abc import ABC, abstractmethod
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

import requests

from engine.security.audit.core import AuditEvent, AuditLevel

logger = logging.getLogger(__name__)


class AlertChannel(ABC):
    """Base class for alert channels."""

    def __init__(self, name: str):
        self.name = name
        self.enabled = False
        self._alert_count = 0
        self._error_count = 0

    @abstractmethod
    def send_alert(
        self,
        event: AuditEvent,
        sanitized_data: Dict[str, Any],
    ) -> bool:
        """Send alert through this channel.

        Args:
            event: The audit event
            sanitized_data: Sanitized event data

        Returns:
            True if alert sent successfully
        """
        pass

    def get_metrics(self) -> Dict[str, Any]:
        """Get channel metrics."""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "alerts_sent": self._alert_count,
            "errors": self._error_count,
        }


class EmailAlertChannel(AlertChannel):
    """Email alerts via SMTP."""

    def __init__(self):
        super().__init__("email")
        self.smtp_host = os.getenv("SMTP_HOST")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_username = os.getenv("SMTP_USERNAME")
        self.smtp_password = os.getenv("SMTP_PASSWORD")
        self.from_address = os.getenv("ALERT_EMAIL_FROM", "audit@eleanor.ai")
        self.to_addresses = os.getenv("ALERT_EMAIL_TO", "").split(",")
        self.use_tls = os.getenv("SMTP_USE_TLS", "true").lower() == "true"

        self.enabled = bool(
            self.smtp_host
            and self.smtp_username
            and self.smtp_password
            and self.to_addresses
        )

    def send_alert(
        self,
        event: AuditEvent,
        sanitized_data: Dict[str, Any],
    ) -> bool:
        """Send email alert."""
        if not self.enabled:
            return False

        try:
            # Create message
            msg = MIMEMultipart("alternative")
            msg["From"] = self.from_address
            msg["To"] = ", ".join(self.to_addresses)
            msg["Subject"] = f"[{event.level.value.upper()}] {event.event_type}"

            # Create plain text and HTML versions
            text_body = self._format_text_body(event, sanitized_data)
            html_body = self._format_html_body(event, sanitized_data)

            msg.attach(MIMEText(text_body, "plain"))
            msg.attach(MIMEText(html_body, "html"))

            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)

            self._alert_count += 1
            return True

        except Exception as exc:
            self._error_count += 1
            logger.warning("email_alert_failed", extra={"error": str(exc)})
            return False

    def _format_text_body(self, event: AuditEvent, data: Dict[str, Any]) -> str:
        """Format plain text email body."""
        lines = [
            f"ELEANOR Audit Alert",
            f"===================",
            f"",
            f"Event Type: {event.event_type}",
            f"Level: {event.level.value.upper()}",
            f"Timestamp: {event.timestamp}",
            f"Event ID: {event.event_id}",
            f"",
        ]

        if event.actor_id:
            lines.append(f"Actor: {event.actor_id}")
        if event.resource_name:
            lines.append(f"Resource: {event.resource_name}")
        if event.action:
            lines.append(f"Action: {event.action}")
        if event.outcome:
            lines.append(f"Outcome: {event.outcome}")

        if event.error_message:
            lines.extend(["", f"Error: {event.error_message}"])

        lines.extend(["", "Full Event Data:", json.dumps(data, indent=2)])

        return "\n".join(lines)

    def _format_html_body(self, event: AuditEvent, data: Dict[str, Any]) -> str:
        """Format HTML email body."""
        color_map = {
            "debug": "#6c757d",
            "info": "#0dcaf0",
            "warning": "#ffc107",
            "error": "#dc3545",
            "critical": "#d63384",
        }
        color = color_map.get(event.level.value, "#0dcaf0")

        return f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <div style="border-left: 4px solid {color}; padding-left: 16px;">
                <h2 style="color: {color};">ELEANOR Audit Alert</h2>
                <p><strong>Event Type:</strong> {event.event_type}</p>
                <p><strong>Level:</strong> {event.level.value.upper()}</p>
                <p><strong>Timestamp:</strong> {event.timestamp}</p>
                <p><strong>Event ID:</strong> {event.event_id}</p>
                {f'<p><strong>Actor:</strong> {event.actor_id}</p>' if event.actor_id else ''}
                {f'<p><strong>Resource:</strong> {event.resource_name}</p>' if event.resource_name else ''}
                {f'<p><strong>Action:</strong> {event.action}</p>' if event.action else ''}
                {f'<p><strong>Outcome:</strong> {event.outcome}</p>' if event.outcome else ''}
                {f'<p style="color: #dc3545;"><strong>Error:</strong> {event.error_message}</p>' if event.error_message else ''}
            </div>
            <details>
                <summary>Full Event Data</summary>
                <pre>{json.dumps(data, indent=2)}</pre>
            </details>
        </body>
        </html>
        """


class SlackAlertChannel(AlertChannel):
    """Slack alerts via webhook."""

    def __init__(self):
        super().__init__("slack")
        self.webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        self.channel = os.getenv("SLACK_CHANNEL")
        self.enabled = bool(self.webhook_url)

    def send_alert(
        self,
        event: AuditEvent,
        sanitized_data: Dict[str, Any],
    ) -> bool:
        """Send Slack alert."""
        if not self.enabled:
            return False

        try:
            color_map = {
                "debug": "#6c757d",
                "info": "#0dcaf0",
                "warning": "#ffc107",
                "error": "#dc3545",
                "critical": "#d63384",
            }

            payload = {
                "text": f"🚨 {event.level.value.upper()}: {event.event_type}",
                "attachments": [
                    {
                        "color": color_map.get(event.level.value, "#0dcaf0"),
                        "fields": [
                            {"title": "Event Type", "value": event.event_type, "short": True},
                            {"title": "Level", "value": event.level.value.upper(), "short": True},
                            {"title": "Timestamp", "value": event.timestamp, "short": True},
                            {"title": "Event ID", "value": event.event_id, "short": True},
                        ],
                    }
                ],
            }

            if event.actor_id:
                payload["attachments"][0]["fields"].append(
                    {"title": "Actor", "value": event.actor_id, "short": True}
                )
            if event.error_message:
                payload["attachments"][0]["fields"].append(
                    {"title": "Error", "value": event.error_message, "short": False}
                )

            if self.channel:
                payload["channel"] = self.channel

            response = requests.post(self.webhook_url, json=payload, timeout=5)
            response.raise_for_status()

            self._alert_count += 1
            return True

        except Exception as exc:
            self._error_count += 1
            logger.warning("slack_alert_failed", extra={"error": str(exc)})
            return False


class PagerDutyAlertChannel(AlertChannel):
    """PagerDuty alerts."""

    def __init__(self):
        super().__init__("pagerduty")
        self.integration_key = os.getenv("PAGERDUTY_INTEGRATION_KEY")
        self.enabled = bool(self.integration_key)

    def send_alert(
        self,
        event: AuditEvent,
        sanitized_data: Dict[str, Any],
    ) -> bool:
        """Send PagerDuty alert."""
        if not self.enabled:
            return False

        try:
            severity_map = {
                "debug": "info",
                "info": "info",
                "warning": "warning",
                "error": "error",
                "critical": "critical",
            }

            payload = {
                "routing_key": self.integration_key,
                "event_action": "trigger",
                "dedup_key": event.event_id,
                "payload": {
                    "summary": f"{event.event_type}: {event.action or 'audit event'}",
                    "severity": severity_map.get(event.level.value, "info"),
                    "source": event.hostname,
                    "timestamp": event.timestamp,
                    "custom_details": sanitized_data,
                },
            }

            response = requests.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=payload,
                timeout=5,
            )
            response.raise_for_status()

            self._alert_count += 1
            return True

        except Exception as exc:
            self._error_count += 1
            logger.warning("pagerduty_alert_failed", extra={"error": str(exc)})
            return False


class TeamsAlertChannel(AlertChannel):
    """Microsoft Teams alerts via webhook."""

    def __init__(self):
        super().__init__("teams")
        self.webhook_url = os.getenv("TEAMS_WEBHOOK_URL")
        self.enabled = bool(self.webhook_url)

    def send_alert(
        self,
        event: AuditEvent,
        sanitized_data: Dict[str, Any],
    ) -> bool:
        """Send Teams alert."""
        if not self.enabled:
            return False

        try:
            color_map = {
                "debug": "808080",
                "info": "0078D7",
                "warning": "FFA500",
                "error": "DC143C",
                "critical": "8B0000",
            }

            facts = [
                {"name": "Event Type", "value": event.event_type},
                {"name": "Level", "value": event.level.value.upper()},
                {"name": "Timestamp", "value": event.timestamp},
                {"name": "Event ID", "value": event.event_id},
            ]

            if event.actor_id:
                facts.append({"name": "Actor", "value": event.actor_id})
            if event.error_message:
                facts.append({"name": "Error", "value": event.error_message})

            payload = {
                "@type": "MessageCard",
                "@context": "https://schema.org/extensions",
                "themeColor": color_map.get(event.level.value, "0078D7"),
                "title": f"🚨 ELEANOR Audit Alert: {event.event_type}",
                "text": f"**{event.level.value.upper()}** level event detected",
                "sections": [{"facts": facts}],
            }

            response = requests.post(self.webhook_url, json=payload, timeout=5)
            response.raise_for_status()

            self._alert_count += 1
            return True

        except Exception as exc:
            self._error_count += 1
            logger.warning("teams_alert_failed", extra={"error": str(exc)})
            return False


class WebhookAlertChannel(AlertChannel):
    """Generic webhook alerts."""

    def __init__(self):
        super().__init__("webhook")
        self.webhook_url = os.getenv("ALERT_WEBHOOK_URL")
        self.webhook_secret = os.getenv("ALERT_WEBHOOK_SECRET")
        self.enabled = bool(self.webhook_url)

    def send_alert(
        self,
        event: AuditEvent,
        sanitized_data: Dict[str, Any],
    ) -> bool:
        """Send webhook alert."""
        if not self.enabled:
            return False

        try:
            headers = {"Content-Type": "application/json"}
            if self.webhook_secret:
                import hmac
                import hashlib

                payload_bytes = json.dumps(sanitized_data).encode("utf-8")
                signature = hmac.new(
                    self.webhook_secret.encode("utf-8"),
                    payload_bytes,
                    hashlib.sha256,
                ).hexdigest()
                headers["X-Webhook-Signature"] = signature

            response = requests.post(
                self.webhook_url,
                json=sanitized_data,
                headers=headers,
                timeout=5,
            )
            response.raise_for_status()

            self._alert_count += 1
            return True

        except Exception as exc:
            self._error_count += 1
            logger.warning("webhook_alert_failed", extra={"error": str(exc)})
            return False


class AlertManager:
    """Manages multiple alert channels."""

    def __init__(self, channels: Optional[List[AlertChannel]] = None):
        self.channels = channels or self._init_channels()
        self._total_alerts = 0

    def _init_channels(self) -> List[AlertChannel]:
        """Initialize all alert channels."""
        channels: List[AlertChannel] = []
        channel_classes = [
            EmailAlertChannel,
            SlackAlertChannel,
            PagerDutyAlertChannel,
            TeamsAlertChannel,
            WebhookAlertChannel,
        ]

        for channel_class in channel_classes:
            try:
                channel = channel_class()
                if channel.enabled:
                    channels.append(channel)
                    logger.info(
                        "alert_channel_enabled",
                        extra={"channel": channel.name},
                    )
            except Exception as exc:
                logger.warning(
                    "alert_channel_init_failed",
                    extra={
                        "channel": channel_class.__name__,
                        "error": str(exc),
                    },
                )

        return channels

    def send_alert(self, event: AuditEvent, sanitized_data: Dict[str, Any]) -> None:
        """Send alert through all configured channels."""
        self._total_alerts += 1

        for channel in self.channels:
            try:
                channel.send_alert(event, sanitized_data)
            except Exception as exc:
                logger.warning(
                    "alert_channel_failed",
                    extra={
                        "channel": channel.name,
                        "event_id": event.event_id,
                        "error": str(exc),
                    },
                )

    def get_metrics(self) -> Dict[str, Any]:
        """Get alert manager metrics."""
        return {
            "total_alerts": self._total_alerts,
            "channels": [channel.get_metrics() for channel in self.channels],
        }


_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get or create the global alert manager."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager


__all__ = [
    "AlertChannel",
    "EmailAlertChannel",
    "SlackAlertChannel",
    "PagerDutyAlertChannel",
    "TeamsAlertChannel",
    "WebhookAlertChannel",
    "AlertManager",
    "get_alert_manager",
]
