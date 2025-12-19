#!/bin/bash
# ELEANOR V8 - Secure Secret Generation Script
# Generates cryptographically secure secrets for production deployment

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🔐 ELEANOR V8 - Secret Generation Tool"
echo "======================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check for required tools
check_requirements() {
    local missing_tools=()

    if ! command -v openssl &> /dev/null; then
        missing_tools+=("openssl")
    fi

    if [ ${#missing_tools[@]} -gt 0 ]; then
        echo -e "${RED}ERROR: Missing required tools: ${missing_tools[*]}${NC}"
        echo "Install with: brew install openssl"
        exit 1
    fi
}

# Generate a secure random secret
generate_secret() {
    local length=${1:-32}
    openssl rand -base64 "$length" | tr -d '\n='
}

# Generate a secure password
generate_password() {
    local length=${1:-20}
    # Generate password with alphanumeric and special characters
    openssl rand -base64 "$length" | tr -d '\n=' | head -c "$length"
}

# Main secret generation
main() {
    check_requirements

    echo -e "${BLUE}Generating secure secrets for ELEANOR V8...${NC}"
    echo ""

    # 1. JWT Secret (32 bytes = 256 bits)
    JWT_SECRET=$(generate_secret 32)
    echo -e "${GREEN}✅ JWT_SECRET (256-bit):${NC}"
    echo "   $JWT_SECRET"
    echo ""

    # 2. Grafana Admin Password
    GRAFANA_PASSWORD=$(generate_password 20)
    echo -e "${GREEN}✅ GRAFANA_ADMIN_PASSWORD:${NC}"
    echo "   $GRAFANA_PASSWORD"
    echo ""

    # 3. PostgreSQL Password
    POSTGRES_PASSWORD=$(generate_password 20)
    echo -e "${GREEN}✅ POSTGRES_PASSWORD:${NC}"
    echo "   $POSTGRES_PASSWORD"
    echo ""

    # 4. Weaviate API Key (if needed for production)
    WEAVIATE_API_KEY=$(generate_secret 32)
    echo -e "${GREEN}✅ WEAVIATE_API_KEY (optional):${NC}"
    echo "   $WEAVIATE_API_KEY"
    echo ""

    echo "======================================="
    echo -e "${YELLOW}⚠️  IMPORTANT SECURITY NOTES:${NC}"
    echo ""
    echo "1. These secrets are displayed ONCE - save them securely"
    echo "2. Never commit these to version control"
    echo "3. Store in a secrets manager (AWS Secrets Manager, HashiCorp Vault, etc.)"
    echo "4. Use different secrets for each environment (dev/staging/prod)"
    echo ""

    # Offer to create .env file
    echo -e "${BLUE}Would you like to create a .env file with these secrets? [y/N]${NC}"
    read -r response

    if [[ "$response" =~ ^[Yy]$ ]]; then
        create_env_file
    else
        echo ""
        echo -e "${YELLOW}Secrets not saved to file. Copy them manually.${NC}"
        echo ""
        echo "Add these to your .env file:"
        echo ""
        echo "JWT_SECRET=$JWT_SECRET"
        echo "GRAFANA_ADMIN_PASSWORD=$GRAFANA_PASSWORD"
        echo "POSTGRES_PASSWORD=$POSTGRES_PASSWORD"
        echo "# WEAVIATE_API_KEY=$WEAVIATE_API_KEY  # Uncomment if using authenticated Weaviate"
    fi
}

# Create .env file with generated secrets
create_env_file() {
    local env_file="$PROJECT_ROOT/.env"

    if [ -f "$env_file" ]; then
        echo ""
        echo -e "${YELLOW}⚠️  .env file already exists!${NC}"
        echo -e "${BLUE}Overwrite existing .env file? [y/N]${NC}"
        read -r overwrite_response

        if [[ ! "$overwrite_response" =~ ^[Yy]$ ]]; then
            echo ""
            echo -e "${YELLOW}Cancelled. Existing .env file preserved.${NC}"
            return
        fi

        # Backup existing .env
        cp "$env_file" "$env_file.backup.$(date +%Y%m%d_%H%M%S)"
        echo -e "${GREEN}✅ Backed up existing .env${NC}"
    fi

    # Copy from .env.sample
    if [ -f "$PROJECT_ROOT/.env.sample" ]; then
        cp "$PROJECT_ROOT/.env.sample" "$env_file"
        echo -e "${GREEN}✅ Copied from .env.sample${NC}"
    else
        echo -e "${RED}ERROR: .env.sample not found${NC}"
        exit 1
    fi

    # Update with generated secrets
    # Use platform-agnostic sed syntax
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s|^ELEANOR_ENV=.*|ELEANOR_ENV=production|" "$env_file"
        sed -i '' "s|^JWT_SECRET=.*|JWT_SECRET=$JWT_SECRET|" "$env_file"
        sed -i '' "s|^GRAFANA_ADMIN_PASSWORD=.*|GRAFANA_ADMIN_PASSWORD=$GRAFANA_PASSWORD|" "$env_file"

        # Add PostgreSQL password if not present
        if ! grep -q "^POSTGRES_PASSWORD=" "$env_file"; then
            echo "" >> "$env_file"
            echo "# PostgreSQL (production)" >> "$env_file"
            echo "POSTGRES_PASSWORD=$POSTGRES_PASSWORD" >> "$env_file"
        else
            sed -i '' "s|^POSTGRES_PASSWORD=.*|POSTGRES_PASSWORD=$POSTGRES_PASSWORD|" "$env_file"
        fi
    else
        # Linux
        sed -i "s|^ELEANOR_ENV=.*|ELEANOR_ENV=production|" "$env_file"
        sed -i "s|^JWT_SECRET=.*|JWT_SECRET=$JWT_SECRET|" "$env_file"
        sed -i "s|^GRAFANA_ADMIN_PASSWORD=.*|GRAFANA_ADMIN_PASSWORD=$GRAFANA_PASSWORD|" "$env_file"

        if ! grep -q "^POSTGRES_PASSWORD=" "$env_file"; then
            echo "" >> "$env_file"
            echo "# PostgreSQL (production)" >> "$env_file"
            echo "POSTGRES_PASSWORD=$POSTGRES_PASSWORD" >> "$env_file"
        else
            sed -i "s|^POSTGRES_PASSWORD=.*|POSTGRES_PASSWORD=$POSTGRES_PASSWORD|" "$env_file"
        fi
    fi

    echo ""
    echo -e "${GREEN}✅ Created $env_file with secure secrets${NC}"
    echo ""
    echo -e "${YELLOW}⚠️  NEXT STEPS:${NC}"
    echo "1. Review $env_file and add your LLM API keys"
    echo "2. Update PG_CONN_STRING with the new POSTGRES_PASSWORD"
    echo "3. Never commit .env to version control (already in .gitignore)"
    echo "4. Run: ./scripts/validate_env.sh to verify configuration"
    echo ""
}

# Run main function
main
