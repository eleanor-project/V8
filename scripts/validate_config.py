#!/usr/bin/env python3
"""
ELEANOR V8 — Configuration Validation Script

Validates configuration for all environments and reports issues.

Usage:
    python scripts/validate_config.py
    python scripts/validate_config.py --env production
    python scripts/validate_config.py --env-file .env.custom
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.config import EleanorSettings, ConfigManager


def validate_environment(env_file: str) -> Dict[str, Any]:
    """
    Validate configuration for a specific environment.
    
    Args:
        env_file: Path to environment file
    
    Returns:
        Validation result dictionary
    """
    result = {
        "environment": env_file,
        "valid": True,
        "warnings": [],
        "errors": [],
        "settings": None,
    }
    
    try:
        # Load settings
        if Path(env_file).exists():
            settings = EleanorSettings(_env_file=env_file)
        else:
            result["errors"].append(f"Environment file not found: {env_file}")
            result["valid"] = False
            return result
        
        result["settings"] = settings.model_dump()
        
        # Validation checks
        manager = ConfigManager()
        validation = manager.validate()
        
        result["warnings"].extend(validation.get("warnings", []))
        result["errors"].extend(validation.get("errors", []))
        
        if not validation["valid"]:
            result["valid"] = False
        
        # Environment-specific checks
        if settings.environment == "production":
            # Production-specific validations
            if settings.precedent.backend == "none":
                result["warnings"].append(
                    "Production without precedent backend configured"
                )
            
            if settings.security.secret_provider == "env":
                result["errors"].append(
                    "Production must use AWS Secrets Manager or Vault, not env vars"
                )
                result["valid"] = False
            
            if not settings.resilience.enable_circuit_breakers:
                result["errors"].append(
                    "Circuit breakers must be enabled in production"
                )
                result["valid"] = False
            
            if not settings.observability.enable_tracing:
                result["warnings"].append(
                    "Tracing should be enabled in production for observability"
                )
            
            if not settings.cache.enabled:
                result["warnings"].append(
                    "Caching should be enabled in production for performance"
                )
        
        # Check required fields based on configuration
        if settings.cache.enabled and not settings.cache.redis_url:
            result["warnings"].append(
                "Caching enabled but no Redis URL configured. "
                "Using in-memory cache only."
            )
        
        if settings.observability.enable_tracing:
            if not settings.observability.otel_endpoint and not settings.observability.jaeger_endpoint:
                result["errors"].append(
                    "Tracing enabled but no endpoint configured"
                )
                result["valid"] = False
        
        if settings.precedent.backend not in ["none", "chroma", "qdrant", "pinecone"]:
            result["errors"].append(
                f"Invalid precedent backend: {settings.precedent.backend}"
            )
            result["valid"] = False
    
    except Exception as e:
        result["valid"] = False
        result["errors"].append(f"Configuration load failed: {str(e)}")
    
    return result


def print_result(result: Dict[str, Any], verbose: bool = False) -> None:
    """Print validation result."""
    env = result["environment"]
    
    print(f"\n{'='*60}")
    print(f"Environment: {env}")
    print(f"{'='*60}")
    
    if result["valid"]:
        print("✅ Configuration is VALID")
    else:
        print("❌ Configuration is INVALID")
    
    # Print errors
    if result["errors"]:
        print("\n❌ ERRORS:")
        for error in result["errors"]:
            print(f"  - {error}")
    
    # Print warnings
    if result["warnings"]:
        print("\n⚠️  WARNINGS:")
        for warning in result["warnings"]:
            print(f"  - {warning}")
    
    # Print settings if verbose
    if verbose and result["settings"]:
        print("\nConfiguration Summary:")
        settings = result["settings"]
        print(f"  Environment: {settings['environment']}")
        print(f"  Detail Level: {settings['detail_level']}")
        print(f"  LLM Provider: {settings['llm']['provider']}")
        print(f"  Precedent Backend: {settings['precedent']['backend']}")
        print(f"  Caching: {'Enabled' if settings['cache']['enabled'] else 'Disabled'}")
        print(f"  Circuit Breakers: {'Enabled' if settings['resilience']['enable_circuit_breakers'] else 'Disabled'}")
        print(f"  Tracing: {'Enabled' if settings['observability']['enable_tracing'] else 'Disabled'}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate ELEANOR configuration"
    )
    parser.add_argument(
        "--env",
        choices=["development", "staging", "production"],
        help="Specific environment to validate"
    )
    parser.add_argument(
        "--env-file",
        help="Specific environment file to validate"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Validate all environments"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    
    args = parser.parse_args()
    
    # Determine which environments to validate
    env_files: List[str] = []
    
    if args.env_file:
        env_files = [args.env_file]
    elif args.env:
        env_files = [f".env.{args.env}"]
    elif args.all:
        env_files = [".env.development", ".env.staging", ".env.production.template"]
    else:
        # Default: validate current environment
        env_files = [".env"]
    
    # Validate each environment
    results = []
    for env_file in env_files:
        result = validate_environment(env_file)
        results.append(result)
    
    # Output results
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for result in results:
            print_result(result, verbose=args.verbose)
        
        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        
        total = len(results)
        valid = sum(1 for r in results if r["valid"])
        invalid = total - valid
        
        print(f"Total environments checked: {total}")
        print(f"✅ Valid: {valid}")
        print(f"❌ Invalid: {invalid}")
        
        if invalid > 0:
            print("\n⚠️  Some environments have configuration issues. Please review above.")
            sys.exit(1)
        else:
            print("\n✅ All configurations are valid!")
            sys.exit(0)


if __name__ == "__main__":
    main()
