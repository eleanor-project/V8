"""
ELEANOR V8 — Configuration Manager

Singleton configuration manager with hot-reload support.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, cast
from threading import Lock

from .settings import EleanorSettings, get_settings

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Singleton configuration manager.

    Provides centralized configuration access with:
    - Environment-aware loading
    - Hot-reload support
    - Configuration validation
    - Legacy format conversion
    """

    _instance: Optional["ConfigManager"] = None
    _lock = Lock()
    _initialized: bool = False

    def __new__(cls) -> "ConfigManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._settings: Optional[EleanorSettings] = None
        self._env_file: Optional[str] = None
        self._initialized = True

        # Load initial configuration
        self.reload()

    def reload(self, env_file: Optional[str] = None) -> None:
        """
        Reload configuration from files.

        Args:
            env_file: Optional specific environment file to load
        """
        if env_file:
            self._env_file = env_file

        try:
            self._settings = get_settings(env_file=self._env_file, reload=True)
            logger.info(
                f"Configuration loaded successfully. " f"Environment: {self._settings.environment}"
            )
            logger.info(
                "Configuration audit",
                extra={"summary": self.audit_summary()},
            )
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    @property
    def settings(self) -> EleanorSettings:
        """Get current settings."""
        if self._settings is None:
            self.reload()
        assert self._settings is not None
        return self._settings

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key.

        Examples:
            config.get("llm.provider")
            config.get("performance.max_concurrency")

        Args:
            key: Dot-notation key
            default: Default value if key not found

        Returns:
            Configuration value
        """
        try:
            parts = key.split(".")
            value = self.settings

            for part in parts:
                value = getattr(value, part)

            return value
        except AttributeError:
            return default

    def validate(self) -> Dict[str, Any]:
        """
        Validate current configuration.

        Returns:
            Validation result with warnings and errors
        """
        result: Dict[str, Any] = {
            "valid": True,
            "warnings": [],
            "errors": [],
        }

        try:
            # Pydantic validation happens on load
            settings = self.settings

            # Additional custom validations
            if settings.environment == "production":
                if settings.precedent.backend == "none":
                    result["warnings"].append("Production without precedent backend")

                if not settings.resilience.enable_circuit_breakers:
                    result["warnings"].append("Circuit breakers disabled in production")

                if settings.security.secret_provider == "env":
                    result["errors"].append("Environment secrets are not allowed in production")
                    result["valid"] = False

            # Check file paths
            evidence_path = Path(settings.evidence.jsonl_path)
            if not evidence_path.parent.exists():
                result["warnings"].append(
                    f"Evidence directory does not exist: {evidence_path.parent}"
                )

        except Exception as e:
            result["valid"] = False
            result["errors"].append(str(e))

        return result

    def audit_summary(self) -> Dict[str, Any]:
        """Return a redacted summary for audit logging."""
        settings = self.settings
        return {
            "environment": settings.environment,
            "detail_level": settings.detail_level,
            "enabled_critics": settings.enabled_critics,
            "precedent_backend": settings.precedent.backend,
            "evidence_path": settings.evidence.jsonl_path,
            "cache_enabled": settings.cache.enabled,
            "tracing_enabled": settings.observability.enable_tracing,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return cast(Dict[str, Any], self.settings.model_dump())

    def to_json(self) -> str:
        """Convert settings to JSON string."""
        return cast(str, self.settings.model_dump_json(indent=2))

    def to_legacy_config(self) -> Dict[str, Any]:
        """Convert to legacy EngineConfig format."""
        return self.settings.to_legacy_engine_config()


# Convenience function for getting config manager
def get_config_manager() -> ConfigManager:
    """Get singleton ConfigManager instance."""
    return ConfigManager()


__all__ = ["ConfigManager", "get_config_manager"]
