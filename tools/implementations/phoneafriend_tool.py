"""
Phone-a-friend tool for consulting an outside model during a conversation.

The tool stores a small, segment-scoped message thread per consulted model so
MIRA can continue with the same outside voice across synchronous tool calls.
"""
import json
import logging
from typing import Any, Dict
from uuid import uuid4

from pydantic import BaseModel, Field

from clients.llm_provider import LLMProvider
from tools.registry import registry
from tools.repo import Tool
from utils.timezone_utils import format_utc_iso, utc_now
from utils.user_context import get_current_segment_id


class PhoneAFriendToolConfig(BaseModel):
    """Configuration for phoneafriend_tool."""

    enabled: bool = Field(
        default=True,
        description="Whether this tool is enabled",
    )


registry.register("phoneafriend_tool", PhoneAFriendToolConfig)


MODEL_SYSTEM_PROMPTS = {
    "claude": """\
You are a level-headed outside thought partner consulted by MIRA through a synchronous tool call.
You do not see MIRA's main context window, conversation history, memories, or system prompt unless the current inquiry includes them.

Give an independent answer to the inquiry. Be calm, precise, and skeptical of weak assumptions.
If the inquiry asks you to continue from earlier phone-a-friend turns, use only this subagent thread's prior messages.
Do not claim access to hidden context. Name uncertainty directly when the inquiry lacks needed facts.""",
    "gemini": """\
You are an outside voice consulted by MIRA through a synchronous tool call.
You have a strong broad understanding of the world, but you do not see MIRA's main context window, conversation history, memories, or system prompt unless the current inquiry includes them.

Use broad world knowledge and clear reasoning to answer the inquiry directly.
If the inquiry asks you to continue from earlier phone-a-friend turns, use only this subagent thread's prior messages.
Do not claim access to hidden context. Name uncertainty directly when the inquiry lacks needed facts.""",
}

MODEL_DESCRIPTIONS = {
    "claude": "level-headed thought partner",
    "gemini": "strong understanding of the world",
}

MODEL_INTERNAL_LLMS = {
    "claude": "phoneafriend_claude",
    "gemini": "phoneafriend_gemini",
}

KEY_PREFIX = "phoneafriend"
THREAD_TTL_SECONDS = 24 * 60 * 60


class PhoneAFriendTool(Tool):
    """Consults an outside model and preserves its thread for the active segment."""

    name = "phoneafriend_tool"
    parallel_safe = False

    simple_description = (
        "Phone another model as an outside voice on an inquiry, with a resumable "
        "segment-scoped subagent thread. claude is a level-headed thought "
        "partner; gemini has a strong understanding of the world."
    )

    anthropic_schema = {
        "name": "phoneafriend_tool",
        "description": (
            "Phone another model as an outside voice to an inquiry. The consulted "
            "model does not see the main context window or conversation history "
            "unless you put that context in inquiry. Reuse subagent_ref to continue "
            "the same outside-model thread for this conversation segment."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "model_choice": {
                    "type": "string",
                    "enum": ["claude", "gemini"],
                    "description": (
                        "Outside model to consult. claude is for having a "
                        "level-headed thought partner. gemini has a very "
                        "strong understanding of the world."
                    ),
                },
                "inquiry": {
                    "type": "string",
                    "description": (
                        "Exact question or request for the outside model. Include "
                        "all context it needs because it cannot see the main "
                        "conversation history, memories, or system prompt."
                    ),
                },
                "subagent_ref": {
                    "type": "string",
                    "description": (
                        "Reconnect string returned by an earlier call, formatted "
                        "as 'phoneafriend:<id>'. Pass it to resume that exact "
                        "outside-model thread. When subagent_ref is provided, omit "
                        "model_choice because the stored thread already fixes the model."
                    ),
                },
            },
            "required": ["inquiry"],
            "additionalProperties": False,
        },
    }

    def __init__(self, llm_provider: LLMProvider | None = None, valkey_client: Any | None = None):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.llm_provider = llm_provider
        self.valkey_client = valkey_client

    def run(
        self,
        inquiry: str,
        model_choice: str | None = None,
        subagent_ref: str | None = None,
    ) -> Dict[str, Any]:
        """Consult or resume an outside model thread for the active segment."""
        inquiry = inquiry.strip() if inquiry else ""
        if not inquiry:
            raise ValueError("inquiry is required")

        segment_id = get_current_segment_id() or "presegment"
        valkey = self._get_valkey()

        if subagent_ref:
            thread = self._load_thread(valkey, segment_id, subagent_ref)
            model_choice = thread["model_choice"]
        else:
            if model_choice is None:
                raise ValueError("model_choice is required when subagent_ref is not provided")
            model_choice = self._validate_model_choice(model_choice)
            thread = self._create_thread(segment_id, model_choice)

        messages = thread["messages"]
        messages.append({"role": "user", "content": inquiry})

        llm_provider = self.llm_provider or LLMProvider()
        response = llm_provider.generate_response(
            messages=list(messages),
            internal_llm=MODEL_INTERNAL_LLMS[model_choice],
            system_override=MODEL_SYSTEM_PROMPTS[model_choice],
        )
        response_text = llm_provider.extract_text_content(response).strip()
        if not response_text:
            raise RuntimeError(f"{model_choice} returned an empty phone-a-friend response")

        messages.append({"role": "assistant", "content": response_text})
        self._save_thread(valkey, thread, messages)

        return {
            "success": True,
            "model_choice": model_choice,
            "model_role": MODEL_DESCRIPTIONS[model_choice],
            "subagent_ref": thread["subagent_ref"],
            "segment_id": segment_id,
            "response": response_text,
            "message": (
                f"{model_choice} responded. Reuse subagent_ref "
                f"{thread['subagent_ref']} to continue this outside-model thread."
            ),
        }

    def _get_valkey(self) -> Any:
        if self.valkey_client is not None:
            return self.valkey_client
        from clients.valkey_client import get_valkey_client

        self.valkey_client = get_valkey_client()
        return self.valkey_client

    def _validate_model_choice(self, model_choice: str) -> str:
        if model_choice not in MODEL_SYSTEM_PROMPTS:
            allowed = ", ".join(MODEL_SYSTEM_PROMPTS)
            raise ValueError(f"model_choice must be one of: {allowed}")
        return model_choice

    def _create_thread(self, segment_id: str, model_choice: str) -> Dict[str, Any]:
        now = format_utc_iso(utc_now())
        thread_id = uuid4().hex
        thread = {
            "thread_id": thread_id,
            "subagent_ref": f"{KEY_PREFIX}:{thread_id}",
            "owner_user_id": self.user_id,
            "segment_id": segment_id,
            "model_choice": model_choice,
            "messages": [],
            "created_at": now,
            "updated_at": now,
        }
        return thread

    def _load_thread(self, valkey: Any, segment_id: str, subagent_ref: str) -> Dict[str, Any]:
        thread_id = self._parse_subagent_ref(subagent_ref)
        key = self._thread_key(segment_id, thread_id)
        raw = valkey.get(key)
        if raw is None:
            raise ValueError(
                f"subagent_ref {subagent_ref} is not active in this conversation segment"
            )
        thread = json.loads(raw)
        if thread.get("owner_user_id") != self.user_id:
            raise ValueError(
                f"subagent_ref {subagent_ref} is not active for the current user"
            )
        if thread.get("segment_id") != segment_id:
            raise ValueError(
                f"subagent_ref {subagent_ref} is not active in this conversation segment"
            )
        self._validate_model_choice(thread["model_choice"])
        return thread

    def _save_thread(
        self,
        valkey: Any,
        thread: Dict[str, Any],
        messages: list[dict[str, str]],
    ) -> None:
        thread["messages"] = messages
        thread["updated_at"] = format_utc_iso(utc_now())
        valkey.setex(
            self._thread_key(thread["segment_id"], thread["thread_id"]),
            THREAD_TTL_SECONDS,
            json.dumps(thread),
        )

    def _parse_subagent_ref(self, subagent_ref: str) -> str:
        prefix = f"{KEY_PREFIX}:"
        if not subagent_ref.startswith(prefix):
            raise ValueError("subagent_ref must be formatted as 'phoneafriend:<id>'")
        thread_id = subagent_ref[len(prefix):].strip()
        if not thread_id:
            raise ValueError("subagent_ref must include a thread id")
        return thread_id

    def _thread_key(self, segment_id: str, thread_id: str) -> str:
        return f"{KEY_PREFIX}:{self.user_id}:{segment_id}:{thread_id}"
