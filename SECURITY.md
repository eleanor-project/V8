# Security Policy

## Supported Versions

- `main` branch is maintained and should receive security fixes. Feature branches may not be hardened.

## Reporting a Vulnerability

- Please email security contacts or open a private GitHub security advisory for this repository.
- Do not open public issues for undisclosed vulnerabilities.

## Secrets Handling

- Never commit secrets. Use `config/secrets.yaml.example` as a template; keep filled secrets files out of version control.
- Environment variables or a secrets manager (AWS Secrets Manager or HashiCorp Vault) are recommended for production.
- Secret scanning: `detect-secrets` baseline is tracked at `.secrets.baseline`; a pre-commit hook (`.git/hooks/pre-commit`) runs `detect-secrets-hook` on staged files. Ensure it is installed in your clone.
- If a secret is leaked:
  1. Rotate the credential immediately.
  2. Purge it from git history (e.g., `git filter-repo`).
  3. Force-push after coordinating with collaborators.
  4. Add/verify gitignore rules for secret files.

## Dependency Security

- Optional advanced features require `pip install .[advanced]`; install only from trusted sources.
- Keep dependencies updated and run security scans (e.g., `pip-audit` or `safety`).

## Disclosure Assistance

- For urgent incidents, pause other work and prioritize rotation, history cleanup, and monitoring.
- Document remediation steps and verify with a clean scan.
