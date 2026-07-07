"""Power-on self-test checks for MIRA startup and live readiness.

The pre-server POST is intentionally run in a subprocess from main.py. That
keeps probe-side singleton initialization out of the serving process and gives
the parent process a hard deadline before Hypercorn can bind a socket.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request as UrlRequest
from urllib.request import urlopen
from uuid import uuid4

from pydantic import BaseModel, Field

from utils.timezone_utils import format_utc_iso, utc_now

logger = logging.getLogger(__name__)

PRE_SERVER_POST_DEADLINE_SECONDS = 300
POST_SERVER_POST_DEADLINE_SECONDS = 60
LLM_PROVIDER_PROBE_MAX_WORKERS = 4
LLM_PROVIDER_PROBE_MAX_TOKENS = 32
POST_PROBE_TEXT = "mira post probe"
REPORT_BEGIN = "MIRA_POST_REPORT_BEGIN"
REPORT_END = "MIRA_POST_REPORT_END"

PostPhase = Literal["pre-server", "post-server"]


class PostCheckResult(BaseModel):
    """Result for one named POST component check."""

    component: str = Field(description="Stable component name for the check")
    required: bool = Field(description="Whether failure blocks the phase")
    passed: bool = Field(description="Whether the component check passed")
    duration_ms: float = Field(description="Wall-clock duration in milliseconds")
    diagnostic: str | None = Field(default=None, description="Failure diagnostic")
    details: dict[str, Any] = Field(default_factory=dict, description="Structured check metadata")


class PostReport(BaseModel):
    """Aggregated POST report for one phase."""

    phase: PostPhase = Field(description="POST phase")
    status: Literal["pass", "fail"] = Field(description="Overall required-check status")
    started_at: str = Field(description="UTC start timestamp")
    duration_ms: float = Field(description="Total phase duration in milliseconds")
    deadline_seconds: float | None = Field(default=None, description="Phase deadline, if enforced")
    checks: list[PostCheckResult] = Field(default_factory=list, description="Component check results")
    required_failures: list[str] = Field(default_factory=list, description="Failed required components")
    advisory_failures: list[str] = Field(default_factory=list, description="Failed advisory components")

    @property
    def required_passed(self) -> bool:
        """True when every required check passed."""
        return not self.required_failures


class PostFailure(RuntimeError):
    """Raised when a POST phase has failed required checks."""

    def __init__(self, report: PostReport):
        self.report = report
        super().__init__(
            f"{report.phase} POST failed required checks: "
            f"{', '.join(report.required_failures)}"
        )


@dataclass(frozen=True)
class CheckSpec:
    """Executable POST check registration."""

    component: str
    required: bool
    probe: Callable[[], dict[str, Any]]


def run_pre_server_post_gate(deadline_seconds: int = PRE_SERVER_POST_DEADLINE_SECONDS) -> PostReport:
    """Run the pre-server POST as a hard startup gate.

    Raises:
        PostFailure: Required checks failed or the child probe exceeded the deadline.
    """
    report = run_pre_server_post_subprocess(deadline_seconds=deadline_seconds)
    report_json = report.model_dump_json(indent=2)

    if report.required_passed:
        logger.info("Pre-server POST passed:\n%s", report_json)
        return report

    logger.critical("Pre-server POST failed:\n%s", report_json)
    raise PostFailure(report)


def run_pre_server_post_subprocess(
    deadline_seconds: int = PRE_SERVER_POST_DEADLINE_SECONDS,
) -> PostReport:
    """Execute the pre-server POST in a child process and parse its report."""
    started = time.monotonic()
    command = [
        sys.executable,
        "-m",
        "utils.power_on_self_test",
        "pre-server",
        "--json",
        "--report-marker",
        "--fast-exit",
        "--deadline-seconds",
        str(deadline_seconds),
    ]

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=deadline_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        report = _parse_report_from_stdout(error.stdout)
        if report is not None:
            if report.required_passed:
                return _append_advisory_failure(
                    report,
                    "pre_server_child_cleanup",
                    (
                        "POST child printed a passing report but did not exit "
                        f"within {deadline_seconds}s"
                    ),
                    {"stderr": _tail_text(error.stderr)},
                )
            return report

        duration_ms = round((time.monotonic() - started) * 1000, 1)
        return _build_report(
            phase="pre-server",
            started_at=format_utc_iso(utc_now()),
            started_monotonic=started,
            deadline_seconds=deadline_seconds,
            checks=[
                PostCheckResult(
                    component="pre_server_deadline",
                    required=True,
                    passed=False,
                    duration_ms=duration_ms,
                    diagnostic=(
                        f"Pre-server POST exceeded {deadline_seconds}s and was killed "
                        "before server bind"
                    ),
                    details={
                        "command": command,
                        "stdout": _tail_text(error.stdout),
                        "stderr": _tail_text(error.stderr),
                    },
                )
            ],
        )

    report = _parse_report_from_stdout(completed.stdout)
    if report is not None:
        if completed.returncode != 0 and report.required_passed:
            return _append_advisory_failure(
                report,
                "pre_server_child_exit",
                (
                    f"POST child exited {completed.returncode} after printing "
                    "a passing report"
                ),
                {"stderr": _tail_text(completed.stderr)},
            )
        return report

    duration_ms = round((time.monotonic() - started) * 1000, 1)
    return _build_report(
        phase="pre-server",
        started_at=format_utc_iso(utc_now()),
        started_monotonic=started,
        deadline_seconds=deadline_seconds,
        checks=[
            PostCheckResult(
                component="pre_server_report_parse",
                required=True,
                passed=False,
                duration_ms=duration_ms,
                diagnostic="Pre-server POST child did not return a valid JSON report",
                details={
                    "returncode": completed.returncode,
                    "stdout": _tail_text(completed.stdout),
                    "stderr": _tail_text(completed.stderr),
                },
            )
        ],
    )


def run_pre_server_post(
    deadline_seconds: float | None = PRE_SERVER_POST_DEADLINE_SECONDS,
) -> PostReport:
    """Run required pre-server checks in the current process."""
    specs = [
        CheckSpec("configuration", True, _check_configuration),
        CheckSpec("vault", True, _check_vault),
        CheckSpec("postgres_rls", True, _check_postgres_rls),
        CheckSpec("valkey", True, _check_valkey),
        CheckSpec("embeddings", True, _check_embeddings),
        CheckSpec("llm_configuration", True, _check_llm_configuration),
        CheckSpec("llm_provider_reachability", True, _check_llm_provider_reachability),
        CheckSpec("tools", True, _check_tools),
        CheckSpec("dependency_initialization", True, _check_dependency_initialization),
        CheckSpec("scheduler_registration", True, _check_scheduler_registration),
        CheckSpec("playwright_service", False, _check_playwright_service),
    ]
    return _run_specs("pre-server", specs, deadline_seconds=deadline_seconds)


def run_post_server_post(
    base_url: str | None = None,
    deadline_seconds: float | None = POST_SERVER_POST_DEADLINE_SECONDS,
) -> PostReport:
    """Run live checks against an already-bound MIRA server."""
    resolved_base_url = base_url or default_post_server_base_url()
    specs = [
        CheckSpec("http_health", True, lambda: _check_http_health(resolved_base_url)),
        CheckSpec("http_diagnostics", True, lambda: _check_http_diagnostics(resolved_base_url)),
    ]
    return _run_specs("post-server", specs, deadline_seconds=deadline_seconds)


def default_post_server_base_url() -> str:
    """Return a loopback URL for the configured API server."""
    from config.config_manager import config

    host = config.api_server.host
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    return f"http://{host}:{config.api_server.port}"


def _run_specs(
    phase: PostPhase,
    specs: list[CheckSpec],
    *,
    deadline_seconds: float | None,
) -> PostReport:
    started_at = format_utc_iso(utc_now())
    started_monotonic = time.monotonic()
    checks: list[PostCheckResult] = []

    for spec in specs:
        if deadline_seconds is not None:
            elapsed = time.monotonic() - started_monotonic
            if elapsed >= deadline_seconds:
                checks.append(
                    PostCheckResult(
                        component=f"{phase.replace('-', '_')}_deadline",
                        required=True,
                        passed=False,
                        duration_ms=round(elapsed * 1000, 1),
                        diagnostic=f"{phase} POST exceeded {deadline_seconds}s",
                    )
                )
                break
        checks.append(_run_one_check(spec))

    return _build_report(
        phase=phase,
        started_at=started_at,
        started_monotonic=started_monotonic,
        deadline_seconds=deadline_seconds,
        checks=checks,
    )


def _run_one_check(spec: CheckSpec) -> PostCheckResult:
    started = time.monotonic()
    try:
        details = spec.probe()
        return PostCheckResult(
            component=spec.component,
            required=spec.required,
            passed=True,
            duration_ms=round((time.monotonic() - started) * 1000, 1),
            details=details,
        )
    except Exception as error:
        return PostCheckResult(
            component=spec.component,
            required=spec.required,
            passed=False,
            duration_ms=round((time.monotonic() - started) * 1000, 1),
            diagnostic=f"{type(error).__name__}: {error}",
            details={"traceback": traceback.format_exc(limit=8)},
        )


def _build_report(
    *,
    phase: PostPhase,
    started_at: str,
    started_monotonic: float,
    deadline_seconds: float | None,
    checks: list[PostCheckResult],
) -> PostReport:
    required_failures = [
        check.component for check in checks if check.required and not check.passed
    ]
    advisory_failures = [
        check.component for check in checks if not check.required and not check.passed
    ]
    return PostReport(
        phase=phase,
        status="fail" if required_failures else "pass",
        started_at=started_at,
        duration_ms=round((time.monotonic() - started_monotonic) * 1000, 1),
        deadline_seconds=deadline_seconds,
        checks=checks,
        required_failures=required_failures,
        advisory_failures=advisory_failures,
    )


def _append_required_failure(
    report: PostReport,
    component: str,
    diagnostic: str,
    details: dict[str, Any],
) -> PostReport:
    checks = list(report.checks)
    checks.append(
        PostCheckResult(
            component=component,
            required=True,
            passed=False,
            duration_ms=0,
            diagnostic=diagnostic,
            details=details,
        )
    )
    return PostReport(
        phase=report.phase,
        status="fail",
        started_at=report.started_at,
        duration_ms=report.duration_ms,
        deadline_seconds=report.deadline_seconds,
        checks=checks,
        required_failures=[*report.required_failures, component],
        advisory_failures=report.advisory_failures,
    )


def _append_advisory_failure(
    report: PostReport,
    component: str,
    diagnostic: str,
    details: dict[str, Any],
) -> PostReport:
    checks = list(report.checks)
    checks.append(
        PostCheckResult(
            component=component,
            required=False,
            passed=False,
            duration_ms=0,
            diagnostic=diagnostic,
            details=details,
        )
    )
    return PostReport(
        phase=report.phase,
        status=report.status,
        started_at=report.started_at,
        duration_ms=report.duration_ms,
        deadline_seconds=report.deadline_seconds,
        checks=checks,
        required_failures=report.required_failures,
        advisory_failures=[*report.advisory_failures, component],
    )


def _parse_report_from_stdout(stdout: str | bytes | None) -> PostReport | None:
    if stdout is None:
        return None
    if isinstance(stdout, bytes):
        candidate = stdout.decode("utf-8", errors="replace").strip()
    else:
        candidate = stdout.strip()
    if not candidate:
        return None
    begin = candidate.rfind(REPORT_BEGIN)
    end = candidate.rfind(REPORT_END)
    if begin != -1 and end != -1 and begin < end:
        candidate = candidate[begin + len(REPORT_BEGIN):end].strip()
    try:
        return PostReport.model_validate_json(candidate)
    except Exception:
        return None


def _tail_text(value: str | bytes | None, limit: int = 4000) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")[-limit:]
    return value[-limit:]


def _check_configuration() -> dict[str, Any]:
    from config.config_manager import config

    if not config.system_prompt.strip():
        raise RuntimeError("System prompt is empty")
    if not isinstance(config.api_server.port, int) or config.api_server.port <= 0:
        raise RuntimeError(f"Invalid API server port: {config.api_server.port}")
    return {
        "api_host": config.api_server.host,
        "api_port": config.api_server.port,
        "workers": config.api_server.workers,
        "provider_response_timeout_seconds": config.api.provider_response_timeout,
        "system_prompt_chars": len(config.system_prompt),
    }


def _check_vault() -> dict[str, Any]:
    from clients.vault_client import (
        VaultClient,
        get_database_url,
        get_service_config,
        preload_secrets,
    )

    vault_client = VaultClient()
    if not vault_client.client.is_authenticated():
        raise PermissionError("Vault client is not authenticated")
    if vault_client.client.sys.is_sealed():
        raise RuntimeError("Vault is sealed")

    preload_secrets()
    service_url = get_database_url("mira_service", admin=False)
    admin_url = get_database_url("mira_service", admin=True)
    valkey_url = get_service_config("valkey_url")

    for label, database_url in {
        "service_url": service_url,
        "admin_url": admin_url,
    }.items():
        parsed = urlparse(database_url)
        if parsed.scheme not in {"postgresql", "postgres"}:
            raise RuntimeError(f"Vault mira/database {label} is not a PostgreSQL URL")
        if not parsed.hostname or not parsed.username:
            raise RuntimeError(f"Vault mira/database {label} is missing host or username")

    parsed_valkey = urlparse(valkey_url)
    if parsed_valkey.scheme not in {"redis", "rediss", "valkey", "valkeys"}:
        raise RuntimeError("Vault mira/services valkey_url is not a Valkey/Redis URL")

    return {
        "vault_addr": vault_client.vault_addr,
        "namespace_configured": bool(vault_client.vault_namespace),
        "database_fields": ["service_url", "admin_url"],
        "service_fields": ["valkey_url"],
    }


def _check_postgres_rls() -> dict[str, Any]:
    from clients.postgres_client import PostgresClient

    service_db = PostgresClient("mira_service")
    admin_db = PostgresClient("mira_service", admin=True)

    service_row = service_db.execute_single(
        """
        SELECT 1 AS ok,
               current_database() AS database_name,
               current_user AS role_name,
               current_setting('app.current_user_id', true) AS current_user_id
        """
    )
    if not service_row or service_row["ok"] != 1:
        raise RuntimeError("Service database SELECT 1 failed")
    if service_row["database_name"] != "mira_service":
        raise RuntimeError(f"Connected to unexpected database {service_row['database_name']}")
    if service_row["current_user_id"] not in ("", None):
        raise RuntimeError("Non-admin pool inherited app.current_user_id")

    admin_row = admin_db.execute_single(
        """
        SELECT current_user AS role_name,
               rolsuper,
               rolbypassrls
        FROM pg_roles
        WHERE rolname = current_user
        """
    )
    if not admin_row or not admin_row["rolbypassrls"]:
        raise RuntimeError("Admin database role does not have BYPASSRLS")

    expected_tables = [
        "continuums",
        "messages",
        "memories",
        "entities",
        "user_activity_days",
        "domain_knowledge_blocks",
        "domain_knowledge_block_content",
        "api_tokens",
        "feedback_signals",
        "feedback_synthesis_tracking",
        "billing_transactions",
    ]
    policy_rows = admin_db.execute_query(
        """
        SELECT c.relname,
               c.relrowsecurity,
               COUNT(p.polname) AS policy_count
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        LEFT JOIN pg_policy p ON p.polrelid = c.oid
        WHERE n.nspname = 'public'
          AND c.relname IN (
              'continuums',
              'messages',
              'memories',
              'entities',
              'user_activity_days',
              'domain_knowledge_blocks',
              'domain_knowledge_block_content',
              'api_tokens',
              'feedback_signals',
              'feedback_synthesis_tracking',
              'billing_transactions'
          )
        GROUP BY c.relname, c.relrowsecurity
        """
    )
    policies_by_table = {row["relname"]: row for row in policy_rows}
    missing_tables = [name for name in expected_tables if name not in policies_by_table]
    rls_disabled = [
        name for name, row in policies_by_table.items() if not row["relrowsecurity"]
    ]
    missing_policies = [
        name for name, row in policies_by_table.items() if row["policy_count"] < 1
    ]
    if missing_tables or rls_disabled or missing_policies:
        raise RuntimeError(
            "RLS metadata invalid: "
            f"missing_tables={missing_tables}, "
            f"rls_disabled={rls_disabled}, "
            f"missing_policies={missing_policies}"
        )

    canary = _run_postgres_rls_canary(admin_db)
    return {
        "service_role": service_row["role_name"],
        "admin_role": admin_row["role_name"],
        "rls_tables_checked": len(expected_tables),
        "rls_canary": canary,
    }


def _run_postgres_rls_canary(admin_db: Any) -> dict[str, Any]:
    from clients.postgres_client import PostgresClient

    owner_id = uuid4()
    other_id = uuid4()
    continuum_id = uuid4()
    suffix = uuid4().hex

    try:
        admin_db.execute_insert(
            """
            INSERT INTO users (id, email, first_name, last_name)
            VALUES (%s, %s, %s, %s), (%s, %s, %s, %s)
            """,
            (
                str(owner_id),
                f"post-owner-{suffix}@mira.local",
                "POST",
                "Owner",
                str(other_id),
                f"post-other-{suffix}@mira.local",
                "POST",
                "Other",
            ),
        )
        admin_db.execute_insert(
            "INSERT INTO continuums (id, user_id, metadata) VALUES (%s, %s, %s)",
            (str(continuum_id), str(owner_id), json.dumps({"post_probe": suffix})),
        )

        owner_db = PostgresClient("mira_service", user_id=str(owner_id))
        other_db = PostgresClient("mira_service", user_id=str(other_id))
        owner_count = owner_db.execute_scalar(
            "SELECT COUNT(*) FROM continuums WHERE id = %s",
            (str(continuum_id),),
        )
        other_count = other_db.execute_scalar(
            "SELECT COUNT(*) FROM continuums WHERE id = %s",
            (str(continuum_id),),
        )

        if owner_count != 1:
            raise RuntimeError("Owner-scoped RLS query could not see its continuum canary")
        if other_count != 0:
            raise RuntimeError("Other-user RLS query could see owner continuum canary")

        return {"owner_visible": owner_count, "other_visible": other_count}
    finally:
        try:
            admin_db.execute_update(
                "DELETE FROM users WHERE id IN (%s, %s)",
                (str(owner_id), str(other_id)),
            )
        except Exception:
            logger.exception("Failed to clean up POST RLS canary users")


def _check_valkey() -> dict[str, Any]:
    from clients.valkey_client import get_valkey_client

    valkey_client = get_valkey_client()
    key = f"post:pre-server:{uuid4()}"
    value = "ok"
    try:
        if not valkey_client.set(key, value, ex=60):
            raise RuntimeError("Valkey SET returned false")
        observed = valkey_client.get(key)
        if observed != value:
            raise RuntimeError("Valkey GET did not return the probe value")
        ttl = valkey_client.ttl(key)
        if ttl <= 0:
            raise RuntimeError(f"Valkey probe key has invalid TTL {ttl}")
        return {
            "host": valkey_client.host,
            "port": valkey_client.port,
            "ttl_seconds": ttl,
        }
    finally:
        valkey_client.delete(key)


def _check_embeddings() -> dict[str, Any]:
    import numpy as np

    from clients.hybrid_embeddings_provider import get_hybrid_embeddings_provider

    provider = get_hybrid_embeddings_provider()
    embedding = provider.encode_realtime(POST_PROBE_TEXT)
    if not isinstance(embedding, np.ndarray):
        raise RuntimeError(f"Embedding result is {type(embedding).__name__}, not ndarray")
    if embedding.shape != (768,):
        raise RuntimeError(f"Embedding shape {embedding.shape} does not match (768,)")
    if embedding.dtype != np.float16:
        raise RuntimeError(f"Embedding dtype {embedding.dtype} does not match float16")
    if not np.isfinite(embedding).all():
        raise RuntimeError("Embedding contains NaN or infinity")

    cache_hit = False
    if provider.query_cache is not None:
        cached = provider.query_cache.get(POST_PROBE_TEXT)
        cache_hit = cached is not None and cached.shape == (768,)
        cache_key = provider.query_cache._get_cache_key(POST_PROBE_TEXT)
        provider.query_cache.valkey.valkey_binary.delete(cache_key)
        if not cache_hit:
            raise RuntimeError("Embedding cache did not return the probe vector")

    return {
        "provider": type(provider).__name__,
        "shape": list(embedding.shape),
        "dtype": str(embedding.dtype),
        "query_cache_verified": cache_hit,
    }


def _check_llm_configuration() -> dict[str, Any]:
    from clients.llm.dialect_registry import get_registry
    from clients.llm.resolver import ModelSelection
    from clients.llm.types import coerce_dialect_name, coerce_effort
    from config.config_manager import config

    registry = get_registry()
    registry.discover()
    rows = _load_llm_config_rows()
    if not rows:
        raise RuntimeError("No LLM configuration rows found")

    visible_conversation = 0
    validated = []
    for row in rows:
        dialect_name = coerce_dialect_name(row["dialect_name"])
        registry.get(dialect_name)
        effort = coerce_effort(row["effort"]) if row.get("effort") else None
        max_tokens = row.get("max_tokens") or config.api.max_tokens
        ModelSelection(
            dialect_name=dialect_name,
            model=row["model"],
            endpoint_url=row.get("endpoint_url"),
            api_key_name=row.get("api_key_name"),
            max_tokens=max_tokens,
            effort=effort,
            internal_llm_name=row["name"] if row["source"] == "internal_llm" else None,
            conversation_llm_name=row["name"] if row["source"] == "conversation_llm" else None,
        )
        if row["source"] == "conversation_llm" and not row.get("hidden"):
            visible_conversation += 1
        validated.append(f"{row['source']}:{row['name']}")

    if visible_conversation == 0:
        raise RuntimeError("No visible conversation LLM configurations found")

    return {
        "rows_validated": len(rows),
        "visible_conversation_llms": visible_conversation,
        "registered_dialects": sorted(registry._classes.keys()),
        "validated_configs": validated,
    }


def _load_llm_config_rows() -> list[dict[str, Any]]:
    from clients.postgres_client import PostgresClient

    db = PostgresClient("mira_service")
    conversation_rows = db.execute_query(
        """
        SELECT 'conversation_llm' AS source,
               name,
               NULL AS tier,
               model,
               endpoint_url,
               api_key_name,
               dialect_name,
               NULL AS effort,
               NULL AS max_tokens,
               hidden
        FROM conversation_llm
        """
    )
    internal_rows = db.execute_query(
        """
        SELECT 'internal_llm' AS source,
               name,
               tier,
               model,
               endpoint_url,
               api_key_name,
               dialect_name,
               effort,
               max_tokens,
               FALSE AS hidden
        FROM internal_llm
        """
    )
    return [*conversation_rows, *internal_rows]


def _check_tools() -> dict[str, Any]:
    from clients.llm.types import ToolDefinition
    from config.config_manager import config
    from tools.repo import ESSENTIAL_TOOLS, ToolRepository

    repo = ToolRepository()
    repo.discover_tools()

    missing_essential = [name for name in ESSENTIAL_TOOLS if name not in repo.tool_classes]
    if missing_essential:
        raise RuntimeError(f"Essential tools missing from discovery: {missing_essential}")

    disabled_essential = []
    for name in ESSENTIAL_TOOLS:
        tool_config = getattr(config, name, None)
        if tool_config is not None and not getattr(tool_config, "enabled", True):
            disabled_essential.append(name)
    if disabled_essential:
        raise RuntimeError(f"Essential tools disabled in config: {disabled_essential}")

    repo.enable_tools_from_config()
    missing_enabled = [name for name in ESSENTIAL_TOOLS if name not in repo.enabled_tools]
    if missing_enabled:
        raise RuntimeError(f"Essential tools not enabled: {missing_enabled}")

    static_schema_count = 0
    dynamic_schema_tools = []
    schema_errors = []
    for name, tool_class in sorted(repo.tool_classes.items()):
        schema_attr = inspect.getattr_static(tool_class, "tool_schema", None)
        if isinstance(schema_attr, dict):
            try:
                ToolDefinition.from_mapping(schema_attr)
                static_schema_count += 1
            except Exception as error:
                schema_errors.append(f"{name}: {error}")
        elif isinstance(schema_attr, property) or schema_attr is None:
            dynamic_schema_tools.append(name)

    if schema_errors:
        raise RuntimeError(f"Tool schema validation failed: {schema_errors}")

    return {
        "discovered_tools": len(repo.tool_classes),
        "enabled_essential_tools": sorted(repo.enabled_tools),
        "static_schemas_validated": static_schema_count,
        "dynamic_schema_tools": dynamic_schema_tools,
    }


def _check_dependency_initialization() -> dict[str, Any]:
    from clients.hybrid_embeddings_provider import get_hybrid_embeddings_provider
    from clients.llm_provider import LLMProvider
    from cns.infrastructure.continuum_pool import get_continuum_pool
    from cns.infrastructure.continuum_repository import get_continuum_repository
    from cns.integration.factory import create_cns_orchestrator
    from cns.services.orchestrator import get_orchestrator, initialize_orchestrator
    from lt_memory.factory import get_lt_memory_factory
    from utils.database_session_manager import get_shared_session_manager
    from utils.user_context import load_internal_llm_configs

    embeddings_provider = get_hybrid_embeddings_provider()
    continuum_repo = get_continuum_repository()
    load_internal_llm_configs()
    lt_memory_factory = get_lt_memory_factory(
        session_manager=get_shared_session_manager(),
        embeddings_provider=embeddings_provider,
        llm_provider=LLMProvider(),
        conversation_repo=continuum_repo,
    )
    orchestrator = create_cns_orchestrator()
    initialize_orchestrator(orchestrator)

    if get_orchestrator() is not orchestrator:
        raise RuntimeError("Global orchestrator getter did not return initialized instance")
    continuum_pool = get_continuum_pool()

    return {
        "embeddings_provider": type(embeddings_provider).__name__,
        "continuum_repository": type(continuum_repo).__name__,
        "lt_memory_factory": type(lt_memory_factory).__name__,
        "orchestrator": type(orchestrator).__name__,
        "continuum_pool": type(continuum_pool).__name__,
        "tool_count": len(orchestrator.tool_repo.tool_classes),
        "working_memory": type(orchestrator.working_memory).__name__,
    }


def _check_scheduler_registration() -> dict[str, Any]:
    from cns.services.orchestrator import get_orchestrator
    from utils.scheduled_tasks import (
        initialize_all_scheduled_tasks,
        register_segment_timeout_job,
        register_sidebar_dispatcher_job,
    )
    from utils.scheduler_service import SchedulerService

    scheduler = SchedulerService()
    initialize_all_scheduled_tasks(scheduler)

    orchestrator = get_orchestrator()
    register_segment_timeout_job(scheduler, orchestrator.event_bus)
    register_sidebar_dispatcher_job(
        scheduler,
        orchestrator.tool_repo,
        orchestrator.event_bus,
    )

    try:
        from billing.drip import DailyDripService

        DailyDripService().register_jobs(scheduler)
    except ImportError:
        pass

    registered = set(scheduler._registered_jobs.keys())
    required = {
        "auth_cleanup",
        "account_garbage_collection",
        "lt_memory_extract_unprocessed_segments",
        "lt_memory_extraction_batch_polling",
        "lt_memory_temporal_score_recalculation",
        "lt_memory_bulk_score_recalculation",
        "lt_memory_batch_cleanup",
        "lt_memory_entity_merge",
        "segment_timeout_detection",
    }
    missing = sorted(required - registered)
    if missing:
        raise RuntimeError(f"Required scheduler jobs not registered: {missing}")

    return {
        "registered_jobs": sorted(registered),
        "job_count": len(registered),
        "scheduler_started": scheduler._running,
    }


def _check_playwright_service() -> dict[str, Any]:
    from utils.playwright_service import PlaywrightService

    instance = PlaywrightService.get_instance()
    return {"service": type(instance).__name__}


def _check_http_health(base_url: str) -> dict[str, Any]:
    payload = _http_get_json(base_url, "/v0/api/health", timeout_seconds=5)
    if not payload.get("success"):
        raise RuntimeError(f"Health endpoint returned unsuccessful payload: {payload}")
    data = payload.get("data") or {}
    if data.get("status") != "healthy":
        raise RuntimeError(f"Health endpoint status is {data.get('status')}")
    components = data.get("components") or {}
    database = components.get("database") or {}
    if database.get("status") != "healthy":
        raise RuntimeError(f"Health endpoint database status is {database.get('status')}")
    return {
        "base_url": base_url,
        "status": data.get("status"),
        "components": sorted(components.keys()),
    }


def _check_http_diagnostics(base_url: str) -> dict[str, Any]:
    from clients.vault_client import get_service_config

    diagnostics_token = get_service_config("diagnostics_token")
    payload = _http_get_json(
        base_url,
        "/v0/api/health/threads",
        timeout_seconds=5,
        headers={"X-MIRA-Diagnostics-Token": diagnostics_token},
    )
    if not payload.get("success"):
        raise RuntimeError(f"Diagnostics endpoint returned unsuccessful payload: {payload}")
    data = payload.get("data") or {}
    if data.get("status") == "critical":
        raise RuntimeError("Diagnostics endpoint reports critical status")

    scheduler = data.get("scheduler")
    if not isinstance(scheduler, dict):
        raise RuntimeError("Diagnostics payload does not include scheduler state")
    if not scheduler.get("running"):
        raise RuntimeError("Scheduler is not running in the live server")

    registered = set(scheduler.get("registered_jobs") or [])
    required = {
        "auth_cleanup",
        "account_garbage_collection",
        "lt_memory_extract_unprocessed_segments",
        "lt_memory_extraction_batch_polling",
        "segment_timeout_detection",
    }
    missing = sorted(required - registered)
    if missing:
        raise RuntimeError(f"Live scheduler missing required jobs: {missing}")

    return {
        "status": data.get("status"),
        "scheduler_running": scheduler.get("running"),
        "registered_jobs": sorted(registered),
        "apscheduler_jobs": scheduler.get("apscheduler_jobs") or [],
    }


def _http_get_json(
    base_url: str,
    path: str,
    *,
    timeout_seconds: int,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    url = urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))
    request = UrlRequest(url, headers=headers or {}, method="GET")
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GET {url} returned HTTP {error.code}: {body}") from error
    except URLError as error:
        raise RuntimeError(f"GET {url} failed: {error}") from error

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"GET {url} did not return JSON: {body[:500]}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"GET {url} returned non-object JSON")
    return payload


def _check_llm_provider_reachability() -> dict[str, Any]:
    rows = _load_llm_config_rows()
    targets = _llm_probe_targets(rows)
    if not targets:
        raise RuntimeError("No active LLM provider targets found")

    successes = []
    failures = []
    max_workers = min(LLM_PROVIDER_PROBE_MAX_WORKERS, len(targets))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_probe_llm_target, target): target for target in targets}
        for future in as_completed(futures):
            target = futures[future]
            try:
                successes.append(future.result())
            except Exception as error:
                failures.append(
                    {
                        "target": _llm_target_label(target),
                        "diagnostic": f"{type(error).__name__}: {error}",
                    }
                )

    if failures:
        raise RuntimeError(f"LLM provider probes failed: {failures}")

    return {
        "targets_probed": len(successes),
        "providers": successes,
    }


def _llm_probe_targets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        if row["source"] == "conversation_llm" and row.get("hidden"):
            continue
        key = (
            row["dialect_name"],
            row.get("endpoint_url") or "",
            row.get("api_key_name") or "",
            row["model"],
        )
        if key not in grouped:
            grouped[key] = dict(row)
            grouped[key]["represented_configs"] = [f"{row['source']}:{row['name']}"]
        else:
            grouped[key]["represented_configs"].append(f"{row['source']}:{row['name']}")
    return list(grouped.values())


def _probe_llm_target(target: dict[str, Any]) -> dict[str, Any]:
    from clients.llm.dialect_registry import get_registry
    from clients.llm.lifecycle import LLMLifecycle
    from clients.llm.resolver import ModelSelection
    from clients.llm.types import Request, RequestMetadata, ThinkingConfig, coerce_dialect_name
    from config.config_manager import config

    dialect_name = coerce_dialect_name(target["dialect_name"])
    max_tokens = min(
        target.get("max_tokens") or config.api.max_tokens,
        LLM_PROVIDER_PROBE_MAX_TOKENS,
    )
    selection = ModelSelection(
        dialect_name=dialect_name,
        model=target["model"],
        endpoint_url=target.get("endpoint_url"),
        api_key_name=target.get("api_key_name"),
        max_tokens=max_tokens,
        effort=None,
        internal_llm_name=target["name"] if target["source"] == "internal_llm" else None,
        conversation_llm_name=target["name"] if target["source"] == "conversation_llm" else None,
    )
    timeout = min(config.api.provider_response_timeout, 15)
    dialect_class = get_registry().get(dialect_name)
    dialect = dialect_class.from_selection(
        selection,
        api_key=None,
        timeout=timeout,
        artifact_sink=None,
    )
    request = Request(
        messages=[{"role": "user", "content": "Reply with OK."}],
        system="MIRA POST provider reachability probe. Reply with OK.",
        model=selection.model,
        max_tokens=max_tokens,
        temperature=0,
        thinking=ThinkingConfig(),
        metadata=RequestMetadata(
            endpoint_url=selection.endpoint_url,
            internal_llm_name=selection.internal_llm_name,
            conversation_llm_name=selection.conversation_llm_name,
        ),
    )
    result = LLMLifecycle(
        accounting_policy=None,
        response_timeout_seconds=timeout,
    ).complete(request, dialect, fallback_factory=None)
    reasoning_text = result.reasoning.text if result.reasoning is not None else ""
    if not result.text.strip() and not reasoning_text.strip() and not result.tool_calls:
        raise RuntimeError("Provider returned no text, reasoning, or tool calls")
    return {
        "target": _llm_target_label(target),
        "dialect_name": dialect_name,
        "model": target["model"],
        "represented_configs": sorted(target.get("represented_configs") or []),
        "response_chars": len(result.text),
        "reasoning_chars": len(reasoning_text),
        "tool_calls": len(result.tool_calls),
        "stop_reason": result.stop_reason,
    }


def _llm_target_label(target: dict[str, Any]) -> str:
    endpoint = target.get("endpoint_url") or target["dialect_name"]
    return f"{target['dialect_name']}:{target['model']}@{endpoint}"


def _print_report(report: PostReport, *, json_output: bool, report_marker: bool = False) -> None:
    if json_output:
        if report_marker:
            print(REPORT_BEGIN)
        print(report.model_dump_json(indent=2))
        if report_marker:
            print(REPORT_END)
        sys.stdout.flush()
        return

    print(f"{report.phase} POST: {report.status}")
    for check in report.checks:
        label = "required" if check.required else "advisory"
        status = "PASS" if check.passed else "FAIL"
        print(f"- {check.component} [{label}]: {status} ({check.duration_ms}ms)")
        if check.diagnostic:
            print(f"  {check.diagnostic}")
    sys.stdout.flush()


def _run_cli_phase(args: argparse.Namespace) -> int:
    try:
        if args.phase == "pre-server":
            report = run_pre_server_post(deadline_seconds=args.deadline_seconds)
        else:
            report = run_post_server_post(
                base_url=args.base_url,
                deadline_seconds=args.deadline_seconds,
            )
        _print_report(report, json_output=args.json, report_marker=args.report_marker)
        exit_code = 0 if report.required_passed else 1
        if getattr(args, "fast_exit", False):
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(exit_code)
        return exit_code
    finally:
        if not getattr(args, "fast_exit", False):
            _close_cli_resources()


def _close_cli_resources() -> None:
    """Close process-local infrastructure created by standalone POST checks."""
    def cleanup_step(label: str, callback: Callable[[], None]) -> None:
        try:
            callback()
        except Exception:
            logger.warning("POST cleanup step failed: %s", label, exc_info=True)

    def cleanup_lt_memory_factory() -> None:
        import lt_memory.factory as factory_module

        instance = getattr(factory_module, "_lt_memory_factory_instance", None)
        if instance is not None:
            instance.cleanup()
            factory_module._lt_memory_factory_instance = None

    def cleanup_orchestrator() -> None:
        import cns.services.orchestrator as orchestrator_module

        instance = getattr(orchestrator_module, "_orchestrator_instance", None)
        if instance is not None and getattr(instance, "event_bus", None) is not None:
            instance.event_bus.shutdown()
        orchestrator_module._orchestrator_instance = None

    def cleanup_session_manager() -> None:
        import utils.database_session_manager as session_module

        manager = getattr(session_module, "_shared_session_manager", None)
        if manager is not None:
            manager.cleanup()
            session_module._shared_session_manager = None

    def cleanup_user_data_managers() -> None:
        from utils.userdata_manager import clear_manager_cache

        clear_manager_cache()

    def cleanup_playwright() -> None:
        from utils.playwright_service import PlaywrightService

        instance = getattr(PlaywrightService, "_instance", None)
        if instance is not None:
            instance.shutdown()
            PlaywrightService._instance = None

    def cleanup_postgres() -> None:
        from clients.postgres_client import PostgresClient

        PostgresClient.close_all_pools()

    cleanup_step("lt_memory_factory", cleanup_lt_memory_factory)
    cleanup_step("orchestrator", cleanup_orchestrator)
    cleanup_step("database_session_manager", cleanup_session_manager)
    cleanup_step("userdata_manager", cleanup_user_data_managers)
    cleanup_step("playwright", cleanup_playwright)
    cleanup_step("postgres", cleanup_postgres)


def main(argv: list[str] | None = None) -> int:
    """CLI for running either POST phase."""
    parser = argparse.ArgumentParser(description="Run MIRA power-on self-test checks")
    parser.add_argument("phase", choices=["pre-server", "post-server"])
    parser.add_argument("--base-url", help="Base URL for post-server HTTP checks")
    parser.add_argument(
        "--deadline-seconds",
        type=float,
        default=None,
        help="Phase deadline in seconds",
    )
    parser.add_argument("--json", action="store_true", help="Print structured JSON report")
    parser.add_argument("--report-marker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--fast-exit", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.deadline_seconds is None:
        args.deadline_seconds = (
            PRE_SERVER_POST_DEADLINE_SECONDS
            if args.phase == "pre-server"
            else POST_SERVER_POST_DEADLINE_SECONDS
        )
    return _run_cli_phase(args)


def post_server_cli(argv: list[str] | None = None) -> int:
    """CLI for scripts/post_server_post.py."""
    parser = argparse.ArgumentParser(description="Run MIRA post-server POST checks")
    parser.add_argument("--base-url", help="Base URL for live MIRA server")
    parser.add_argument(
        "--deadline-seconds",
        type=float,
        default=POST_SERVER_POST_DEADLINE_SECONDS,
        help="Phase deadline in seconds",
    )
    parser.add_argument("--json", action="store_true", help="Print structured JSON report")
    args = parser.parse_args(argv)
    try:
        report = run_post_server_post(
            base_url=args.base_url,
            deadline_seconds=args.deadline_seconds,
        )
        _print_report(report, json_output=args.json)
        return 0 if report.required_passed else 1
    finally:
        _close_cli_resources()


if __name__ == "__main__":
    raise SystemExit(main())
