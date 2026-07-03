# Offline Model Preparation

Offline mode routes chat and internal LLM calls to local OpenAI-compatible endpoints instead of hosted providers. The installer configures MIRA for those endpoints, but the local model servers and model files must exist before first startup.

## Default Local Endpoints

The deploy scripts expect:

- Main model server: `http://localhost:3090/v1/chat/completions`
- Small model server: `http://localhost:3092/v1/chat/completions`
- Health check: `curl http://localhost:3090/health`

Default model storage:

```text
/opt/mira/models/
```

Default llama-server logs:

```text
/opt/mira/logs/llama-main.log
/opt/mira/logs/llama-small.log
```

## Model Expectations

The automatic offline profile is designed around two local llama-server instances:

- Main chat model for primary conversation work
- Smaller model for lower-cost internal analysis and maintenance tasks

If you bring your own GGUF models, keep the endpoint URLs and model names aligned with the values selected during `deploy/deploy.sh` configuration. MIRA treats those endpoints as OpenAI-compatible dialects.

## Startup Checklist

Before starting MIRA in offline mode:

1. Place the GGUF model files under `/opt/mira/models/` or the path you configured.
2. Start the main llama-server on port `3090`.
3. Start the small llama-server on port `3092`.
4. Confirm both endpoints respond before launching MIRA.

The application will fail loudly if a required local provider is not reachable.
