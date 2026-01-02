#!/bin/bash
# ELEANOR V8 - Environment Configuration Validator
# Validates production configuration security requirements

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_ROOT/.env"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Counters
ERRORS=0
WARNINGS=0
PASSED=0

# Logging functions
log_error() {
    echo -e "${RED}❌ ERROR: $1${NC}"
    ((ERRORS++))
}

log_warning() {
    echo -e "${YELLOW}⚠️  WARNING: $1${NC}"
    ((WARNINGS++))
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
    ((PASSED++))
}

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if .env file exists
check_env_file_exists() {
    echo ""
    echo "1. Checking .env file existence..."
    if [ ! -f "$ENV_FILE" ]; then
        log_error ".env file not found at $ENV_FILE"
        echo ""
        echo "Create one by running:"
        echo "  ./scripts/generate_secrets.sh"
        echo "Or copy from sample:"
        echo "  cp .env.sample .env"
        exit 1
    else
        log_success ".env file exists"
    fi
}

# Load environment variables from .env
load_env() {
    if [ -f "$ENV_FILE" ]; then
        set -a
        # shellcheck disable=SC1090
        source "$ENV_FILE"
        set +a
    fi
}

# Validate ELEANOR_ENV
check_eleanor_env() {
    echo ""
    echo "2. Checking ELEANOR_ENVIRONMENT setting..."

    local env_value="${ELEANOR_ENVIRONMENT:-${ELEANOR_ENV:-}}"

    if [ -z "$env_value" ]; then
        log_error "ELEANOR_ENVIRONMENT (or ELEANOR_ENV) is not set"
        echo "   Set to 'production' for production deployment"
    elif [ "$env_value" = "development" ]; then
        log_warning "ELEANOR_ENVIRONMENT is set to 'development'"
        echo "   For production, set: ELEANOR_ENVIRONMENT=production"
    elif [ "$env_value" = "production" ]; then
        log_success "ELEANOR_ENVIRONMENT=production (correct for production)"
    else
        log_warning "ELEANOR_ENVIRONMENT='$env_value' (unexpected value)"
        echo "   Expected: 'production' or 'development'"
    fi
}

# Validate JWT_SECRET
check_jwt_secret() {
    echo ""
    echo "3. Checking JWT_SECRET..."

    if [ -z "${JWT_SECRET:-}" ]; then
        log_error "JWT_SECRET is not set"
        echo "   Generate with: openssl rand -base64 32"
    elif [ ${#JWT_SECRET} -lt 32 ]; then
        log_error "JWT_SECRET is too short (${#JWT_SECRET} chars, minimum 32)"
        echo "   Generate a stronger secret: openssl rand -base64 32"
    elif [ "$JWT_SECRET" = "your-secret-key-here" ] || [ "$JWT_SECRET" = "dev-secret" ]; then
        log_error "JWT_SECRET is using a default/insecure value"
        echo "   Generate with: openssl rand -base64 32"
    else
        log_success "JWT_SECRET is set and sufficiently long (${#JWT_SECRET} chars)"
    fi
}

# Validate LLM API Keys
check_llm_keys() {
    echo ""
    echo "4. Checking LLM API Keys..."

    local has_key=false

    if [ -n "${OPENAI_KEY:-}" ] && [ "$OPENAI_KEY" != "your-openai-key" ]; then
        log_success "OPENAI_KEY is configured"
        has_key=true
    fi

    if [ -n "${ANTHROPIC_KEY:-}" ] && [ "$ANTHROPIC_KEY" != "your-anthropic-key" ]; then
        log_success "ANTHROPIC_KEY is configured"
        has_key=true
    fi

    if [ -n "${XAI_KEY:-}" ] && [ "$XAI_KEY" != "your-xai-key" ]; then
        log_success "XAI_KEY is configured"
        has_key=true
    fi

    if [ -n "${GEMINI_KEY:-}" ] && [ "$GEMINI_KEY" != "your-gemini-key" ]; then
        log_success "GEMINI_KEY is configured"
        has_key=true
    fi

    if [ "$has_key" = false ]; then
        log_warning "No LLM API keys configured"
        echo "   At least one of: OPENAI_KEY, ANTHROPIC_KEY, XAI_KEY, GEMINI_KEY"
        echo "   Or configure Ollama for local models"
    fi
}

# Validate Grafana credentials
check_grafana_credentials() {
    echo ""
    echo "5. Checking Grafana credentials..."

    if [ -z "${GRAFANA_ADMIN_PASSWORD:-}" ]; then
        log_warning "GRAFANA_ADMIN_PASSWORD not set in .env"
        echo "   Default 'admin' will be used (check docker-compose.yaml)"
    elif [ "$GRAFANA_ADMIN_PASSWORD" = "admin" ]; then
        log_error "GRAFANA_ADMIN_PASSWORD is using default 'admin'"
        echo "   Generate secure password: openssl rand -base64 20"
    else
        log_success "GRAFANA_ADMIN_PASSWORD is set to custom value"
    fi

    if [ -n "${GRAFANA_ADMIN_USER:-}" ] && [ "$GRAFANA_ADMIN_USER" != "admin" ]; then
        log_success "GRAFANA_ADMIN_USER is customized"
    else
        log_info "GRAFANA_ADMIN_USER using default 'admin' (optional to change)"
    fi
}

# Validate database credentials
check_database_credentials() {
    echo ""
    echo "6. Checking database credentials..."

    # PostgreSQL
    if [ -n "${POSTGRES_PASSWORD:-}" ]; then
        if [ "$POSTGRES_PASSWORD" = "postgres" ]; then
            log_error "POSTGRES_PASSWORD is using default 'postgres'"
            echo "   Generate secure password: openssl rand -base64 20"
        else
            log_success "POSTGRES_PASSWORD is set to custom value"
        fi
    else
        log_warning "POSTGRES_PASSWORD not set (default 'postgres' from docker-compose)"
    fi

    # Check PG_CONN_STRING matches password
    if [ -n "${PG_CONN_STRING:-}" ] && [ -n "${POSTGRES_PASSWORD:-}" ]; then
        if [[ "$PG_CONN_STRING" == *"postgres:postgres@"* ]] && [ "$POSTGRES_PASSWORD" != "postgres" ]; then
            log_warning "PG_CONN_STRING still uses default password"
            echo "   Update to: postgresql://postgres:$POSTGRES_PASSWORD@pgvector:5432/eleanor"
        fi
    fi
}

# Validate CORS configuration
check_cors() {
    echo ""
    echo "7. Checking CORS configuration..."

    if [ -z "${CORS_ORIGINS:-}" ]; then
        log_warning "CORS_ORIGINS not set"
        echo "   Set to production domains, e.g.: https://app.yourdomain.com"
    elif [[ "$CORS_ORIGINS" == *"*"* ]]; then
        log_error "CORS_ORIGINS contains wildcard (*) - security risk"
        echo "   Use specific origins: https://app.yourdomain.com,https://admin.yourdomain.com"
    elif [[ "$CORS_ORIGINS" == *"localhost"* ]]; then
        log_warning "CORS_ORIGINS contains 'localhost'"
        echo "   For production, use production domain names only"
    else
        log_success "CORS_ORIGINS configured without wildcards or localhost"
    fi
}

# Validate OPA configuration
check_opa_config() {
    echo ""
    echo "8. Checking OPA governance configuration..."

    if [ "${ELEANOR_DISABLE_OPA:-0}" = "1" ] || [ "${ELEANOR_DISABLE_OPA:-0}" = "true" ]; then
        log_error "OPA is DISABLED (ELEANOR_DISABLE_OPA=1)"
        echo "   OPA governance is required for production"
        echo "   Set: ELEANOR_DISABLE_OPA=0"
    else
        log_success "OPA is enabled"
    fi

    if [ "${OPA_FAIL_STRATEGY:-escalate}" = "allow" ]; then
        log_error "OPA_FAIL_STRATEGY=allow (security risk)"
        echo "   Use 'escalate' or 'deny' for production"
    elif [ "${OPA_FAIL_STRATEGY:-escalate}" = "escalate" ]; then
        log_success "OPA_FAIL_STRATEGY=escalate (recommended)"
    else
        log_info "OPA_FAIL_STRATEGY=${OPA_FAIL_STRATEGY:-escalate}"
    fi
}

# Validate precedent backend
check_precedent_backend() {
    echo ""
    echo "9. Checking precedent storage configuration..."

    local backend="${PRECEDENT_BACKEND:-memory}"

    if [ "$backend" = "memory" ]; then
        log_warning "PRECEDENT_BACKEND=memory (data lost on restart)"
        echo "   For production, use 'weaviate' or 'pgvector'"
    elif [ "$backend" = "weaviate" ]; then
        log_success "PRECEDENT_BACKEND=weaviate (persistent storage)"
    elif [ "$backend" = "pgvector" ]; then
        log_success "PRECEDENT_BACKEND=pgvector (persistent storage)"
    else
        log_warning "PRECEDENT_BACKEND=$backend (unknown backend)"
    fi
}

# Validate monitoring configuration
check_monitoring() {
    echo ""
    echo "10. Checking monitoring configuration..."

    if [ "${ENABLE_PROMETHEUS_MIDDLEWARE:-1}" = "1" ]; then
        log_success "Prometheus middleware enabled"
    else
        log_warning "Prometheus middleware disabled"
        echo "   Recommended for production monitoring"
    fi

    if [ "${ENABLE_OTEL:-0}" = "1" ]; then
        if [ -z "${OTEL_EXPORTER_OTLP_ENDPOINT:-}" ]; then
            log_warning "OTEL enabled but OTEL_EXPORTER_OTLP_ENDPOINT not set"
        else
            log_success "OpenTelemetry configured"
        fi
    fi
}

# Validate audit logging
check_audit_logging() {
    echo ""
    echo "11. Checking audit logging configuration..."

    if [ -n "${EVIDENCE_PATH:-}" ]; then
        log_success "EVIDENCE_PATH configured: $EVIDENCE_PATH"
    else
        log_info "EVIDENCE_PATH not set (will use default)"
    fi

    if [ -n "${REPLAY_LOG_PATH:-}" ]; then
        log_success "REPLAY_LOG_PATH configured: $REPLAY_LOG_PATH"
    else
        log_info "REPLAY_LOG_PATH not set (will use default)"
    fi
}

# Check for common security misconfigurations
check_security_misconfigurations() {
    echo ""
    echo "12. Checking for security misconfigurations..."

    # Check for debug mode
    if grep -qi "debug.*true" "$ENV_FILE" 2>/dev/null; then
        log_warning "Debug mode may be enabled (found 'debug' in .env)"
    fi

    # Check for test/dev API keys
    if grep -E "test|dev|fake|example|your-.*-key" "$ENV_FILE" 2>/dev/null | grep -v "^#" | grep -v "ELEANOR_ENV"; then
        log_warning "Possible test/dev API keys detected"
        echo "   Review and replace with production keys"
    fi

    # Check file permissions
    if [ -f "$ENV_FILE" ]; then
        local perms=$(stat -f "%Lp" "$ENV_FILE" 2>/dev/null || stat -c "%a" "$ENV_FILE" 2>/dev/null)
        if [ "$perms" != "600" ] && [ "$perms" != "400" ]; then
            log_warning ".env file permissions are $perms (recommend 600 or 400)"
            echo "   Run: chmod 600 $ENV_FILE"
        else
            log_success ".env file has secure permissions ($perms)"
        fi
    fi
}

# Print summary
print_summary() {
    echo ""
    echo "========================================"
    echo "VALIDATION SUMMARY"
    echo "========================================"
    echo -e "${GREEN}Passed:  $PASSED${NC}"
    echo -e "${YELLOW}Warnings: $WARNINGS${NC}"
    echo -e "${RED}Errors:   $ERRORS${NC}"
    echo ""

    if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
        echo -e "${GREEN}✅ Configuration is production-ready!${NC}"
        exit 0
    elif [ $ERRORS -eq 0 ]; then
        echo -e "${YELLOW}⚠️  Configuration has warnings but is acceptable${NC}"
        echo "Review warnings above and address if needed"
        exit 0
    else
        echo -e "${RED}❌ Configuration has ERRORS that must be fixed${NC}"
        echo ""
        echo "Fix the errors above before deploying to production"
        echo ""
        echo "Run ./scripts/generate_secrets.sh to generate secure secrets"
        exit 1
    fi
}

# Main execution
main() {
    echo "🔐 ELEANOR V8 - Environment Configuration Validator"
    echo "===================================================="

    check_env_file_exists
    load_env

    check_eleanor_env
    check_jwt_secret
    check_llm_keys
    check_grafana_credentials
    check_database_credentials
    check_cors
    check_opa_config
    check_precedent_backend
    check_monitoring
    check_audit_logging
    check_security_misconfigurations

    print_summary
}

main
