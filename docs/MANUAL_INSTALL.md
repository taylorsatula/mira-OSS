# Manual Installation

The preferred installation path is still `deploy/deploy.sh`. Use this document when the installer rejects the platform or when you need to reproduce its steps manually.

## Supported Automation

The installer supports:

- macOS with Homebrew
- Debian/Ubuntu-family Linux with `apt`
- Fedora/RHEL/CentOS/Rocky/Alma-family Linux with `dnf`

Other platforms need equivalent services and packages installed manually.

## Required Services

Install and start:

- PostgreSQL 17 with the `pgvector` extension
- Valkey
- HashiCorp Vault
- Python 3.12

The default local ports are:

- MIRA HTTP: `1993`
- Vault: `8200`
- Valkey: `6379`
- PostgreSQL: `5432`

## Python Environment

From the repository root:

```bash
python3.12 -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt
```

After installing Python dependencies, download the spaCy model:

```bash
venv/bin/python -m spacy download en_core_web_lg
```

For web rendering support, install the Playwright browser:

```bash
venv/bin/playwright install chromium
```

## Database

Create the `mira_service` database and load the current schema:

```bash
createdb mira_service
psql -U postgres -h localhost -d mira_service -f deploy/mira_service_schema.sql
```

Vault stores service credentials and provider keys. The deploy scripts in `deploy/vault.sh` and `deploy/postgresql.sh` are the source of truth for the exact key names used by the automated path.

## Running

Once services, credentials, schema, and Python dependencies are in place:

```bash
venv/bin/python main.py
```

Use `deploy/deploy.sh --migrate` for existing automated installations that need a supported upgrade path.
