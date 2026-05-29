"""
Image generation tool via Google Gemini (Nano Banana 2) with Chat-based lineage.

Uses persistent Chat sessions for refinement chains: generate creates a new
session, refine replays the full curated history (all prior prompts + images)
so the model has complete visual+textual context. publish emits the final image
to the user.
"""

import base64
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from uuid import uuid4

from pydantic import BaseModel, Field
from tools.repo import Tool
from tools.registry import registry
from utils.timezone_utils import utc_now


class ImageGenerationToolConfig(BaseModel):
    """Configuration for imagegen_tool."""
    enabled: bool = Field(default=True, description="Whether this tool is enabled by default")


registry.register("imagegen_tool", ImageGenerationToolConfig)

MODEL = "gemini-3.1-flash-image-preview"


class ImageGenerationTool(Tool):
    """Image generation and refinement via Google Gemini (Nano Banana 2).

    Uses Chat sessions with serialized history so refinement chains carry
    full prompt + image lineage automatically.
    """

    name = "imagegen_tool"
    simple_description = "Generate, refine, and publish images using AI. Supports iterative refinement with full prompt and image context across edits."
    description = "Generate, refine, and publish images using AI with full refinement lineage"
    parallel_safe = False

    tool_schema = {
        "name": "imagegen_tool",
        "description": (
            "Generate and refine AI images. Results return to you only — "
            "the user sees the image when you publish it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["generate", "refine", "publish"],
                    "description": (
                        "generate = create image from prompt (returned to you only, user does NOT see it). "
                        "refine = modify a previously generated image by image_id (returned to you only, user does NOT see it). "
                        "publish = send the image to the user by image_id (user sees it for the first time)."
                    )
                },
                "prompt": {
                    "type": "string",
                    "description": "Text prompt describing the image to create. Required for generate. If style_notes is provided, appended as '\\nStyle: {style_notes}'"
                },
                "image_id": {
                    "type": "string",
                    "description": "imggen_XXXXXXXXXXXX identifier returned by a previous generate or refine call. Required for refine and publish"
                },
                "instructions": {
                    "type": "string",
                    "description": "What to change in the existing image. The model sees all prior prompts and images in the refinement chain. Required for refine"
                },
                "preserve": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Literal element names appended to instructions as "
                        "'Preserve these elements unchanged: {comma-separated list}'. "
                        "refine only. Example: [\"background color\", \"character pose\"]"
                    )
                },
                "style_notes": {
                    "type": "string",
                    "description": (
                        "Style directive appended to prompt on generate as '\\nStyle: {value}'. "
                        "Persists automatically across refinements — no need to repeat on refine. "
                        "generate only. Example: \"watercolor\", \"pixel art\", \"photorealistic\""
                    )
                },
                "caption": {
                    "type": "string",
                    "description": "Alt text shown to the user alongside the published image. Default: 'Generated image'. publish only"
                },
                "aspect_ratio": {
                    "type": "string",
                    "enum": [
                        "1:1", "3:4", "4:3", "9:16", "16:9", "3:2", "2:3",
                        "1:4", "1:8", "4:1", "4:5", "5:4", "8:1", "21:9"
                    ],
                    "description": "Image aspect ratio applied per-call. On refine, defaults to the ratio from the previous generate/refine if omitted. Default on generate: 1:1"
                },
                "image_size": {
                    "type": "string",
                    "enum": ["512px", "1K", "2K", "4K"],
                    "description": "Output resolution applied per-call on generate and refine. Higher values produce more detail but take longer. Default: 1K"
                },
            },
            "required": ["operation"]
        }
    }

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self._genai_client = None

    @property
    def genai_client(self):
        """Lazy-init Google GenAI client with per-user API key from UserCredentialService."""
        if self._genai_client is None:
            from google import genai
            from utils.user_credentials import UserCredentialService

            cred_service = UserCredentialService()
            api_key = cred_service.get_credential('api_key', 'google_genai')
            if not api_key:
                raise ValueError(
                    "Google GenAI API key not found. "
                    "Add it in Settings > API Credentials with service name 'google_genai'."
                )
            self._genai_client = genai.Client(api_key=api_key)
        return self._genai_client

    def _ensure_generations_table(self):
        """Create image_generations table if it doesn't exist.

        Called on every operation since tool instances are not cached
        (ToolRepository creates fresh instances per invocation).
        CREATE TABLE IF NOT EXISTS is a no-op on existing tables.
        """
        schema = """
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            file_id TEXT NOT NULL,
            encrypted__prompt TEXT NOT NULL,
            aspect_ratio TEXT DEFAULT '1:1',
            created_at TEXT NOT NULL
        """
        self.db.create_table('image_generations', schema)

    def _sessions_dir(self) -> Path:
        """Return the sessions directory, creating it if needed."""
        sessions = self.user_data_path / "sessions"
        sessions.mkdir(parents=True, exist_ok=True)
        return sessions

    def _save_session(self, session_id: str, chat) -> None:
        """Serialize Chat history to JSON file.

        Base64-encodes inline_data.data bytes for JSON compatibility.
        _load_session_history() reverses this on load.
        """
        history = chat.get_history(curated=True)
        data = [content.model_dump(mode='json') for content in history]
        session_path = self._sessions_dir() / f"{session_id}.json"
        session_path.write_text(json.dumps(data))

    def _load_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Load serialized Chat history from JSON file.

        Returns dicts suitable for types.Content.model_validate() — Pydantic
        handles base64 string → bytes conversion for inline_data.data fields.
        """
        session_path = self._sessions_dir() / f"{session_id}.json"
        if not session_path.exists():
            raise ValueError(f"Session history not found for session: {session_id}")
        return json.loads(session_path.read_text())

    def _create_chat(self, history: List[Any] | None = None):
        """Create a Google GenAI Chat session with image generation modalities.

        Only sets response_modalities at Chat level. Image parameters
        (aspect_ratio, image_size) are per-message — see _message_config().
        """
        from google.genai import types

        config = types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
        )

        kwargs: Dict[str, Any] = {"model": MODEL, "config": config}
        if history is not None:
            kwargs["history"] = history

        return self.genai_client.chats.create(**kwargs)

    def _message_config(
        self,
        aspect_ratio: str = "1:1",
        image_size: str = "1K",
    ) -> "types.GenerateContentConfig":
        """Build per-message config with image parameters.

        Passed to chat.send_message(config=...) so each turn in a
        refinement chain can have different aspect ratio and resolution.
        """
        from google.genai import types

        return types.GenerateContentConfig(
            image_config=types.ImageConfig(
                aspect_ratio=aspect_ratio,
                image_size=image_size,
            ),
        )

    def _save_image(self, image_bytes: bytes, mime_type: str) -> str:
        """Save image bytes to user's tmp directory following the .bin + .meta sidecar convention.

        Returns:
            file_id (str): Identifier for retrieving the image later.
        """
        from utils.user_context import get_current_user_id

        user_id = str(get_current_user_id())
        file_id = f"imggen_{uuid4().hex[:12]}"
        file_dir = Path("data/users") / user_id / "artifacts" / file_id
        file_dir.mkdir(parents=True, exist_ok=True)

        random_stem = uuid4().hex
        content_path = file_dir / f"{random_stem}.bin"
        content_path.write_bytes(image_bytes)

        ext = "png" if "png" in mime_type else "jpeg" if "jpeg" in mime_type else "webp"
        meta_path = file_dir / f"{random_stem}.meta"
        meta_path.write_text(json.dumps({
            "filename": f"generated_image.{ext}",
            "mime_type": mime_type,
        }))

        self.logger.info(f"Saved generated image: {file_id} ({len(image_bytes)} bytes, {mime_type})")
        return file_id

    def _load_image(self, file_id: str) -> tuple[bytes, str]:
        """Load image bytes and mime_type from the user's artifacts directory.

        Returns:
            Tuple of (image_bytes, mime_type)

        Raises:
            ValueError: If file_id not found or directory structure is invalid.
        """
        from utils.user_context import get_current_user_id

        user_id = str(get_current_user_id())
        file_dir = Path("data/users") / user_id / "artifacts" / file_id

        if not file_dir.is_dir():
            raise ValueError(f"Image not found: {file_id}")

        bin_files = list(file_dir.glob("*.bin"))
        meta_files = list(file_dir.glob("*.meta"))
        if not bin_files or not meta_files:
            raise ValueError(f"Image data incomplete for: {file_id}")

        image_bytes = bin_files[0].read_bytes()
        meta = json.loads(meta_files[0].read_text())
        mime_type = meta.get("mime_type", "image/png")

        return image_bytes, mime_type

    def _build_content_blocks(
        self, image_bytes: bytes, mime_type: str, metadata: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """Build rich content blocks with compressed image + metadata.

        Compresses to 512px WebP (same storage tier as user uploads) so the
        image is small enough to persist in conversation history and replay
        on subsequent turns. Full-res stays on disk for the publish path.
        """
        from utils.image_compression import compress_image

        compressed = compress_image(image_bytes, mime_type)
        return [
            {
                "type": "image",
                "media_type": "image/webp",
                "data": compressed.storage_base64,
            },
            {
                "type": "text",
                "text": json.dumps(metadata),
            }
        ]

    def _extract_image_from_response(self, response) -> tuple[bytes, str, str]:
        """Extract image bytes and any text from a Gemini response.

        Works for both direct generate_content() and Chat send_message() responses
        since they share the same GenerateContentResponse structure.

        Returns:
            Tuple of (image_bytes, mime_type, response_text)

        Raises:
            ValueError: If no image was generated in the response.
        """
        image_bytes = None
        mime_type = "image/png"
        response_text = ""

        if not response.candidates or not response.candidates[0].content.parts:
            raise ValueError("No content generated. The prompt may have been filtered.")

        for part in response.candidates[0].content.parts:
            if part.text is not None:
                response_text = part.text
            elif part.inline_data is not None:
                image_bytes = part.inline_data.data
                mime_type = part.inline_data.mime_type or "image/png"

        if image_bytes is None:
            raise ValueError(
                "No image was generated. The model returned text only. "
                "Try a more descriptive prompt or a different aspect ratio."
            )

        return image_bytes, mime_type, response_text

    def _lookup_generation(self, image_id: str) -> Dict[str, Any]:
        """Look up a generation record by image_id.

        Raises:
            ValueError: If image_id not found — includes guidance for legacy references.
        """
        self._ensure_generations_table()
        rows = self.db.select('image_generations', where='id = :id', params={'id': image_id})
        if not rows:
            raise ValueError(
                f"Image ID '{image_id}' not found. This may reference an image from before "
                "the lineage update — generate a new image to start a refinement chain."
            )
        return rows[0]

    def run(self, **params) -> Dict[str, Any]:
        """Route to the appropriate operation handler."""
        operation = params.pop("operation", None)
        if not operation:
            raise ValueError("Required parameter 'operation' not provided")

        if operation == "generate":
            return self._generate(**params)
        elif operation == "refine":
            return self._refine(**params)
        elif operation == "publish":
            return self._publish(**params)
        else:
            raise ValueError(
                f"Unknown operation: {operation}. Must be: generate, refine, or publish"
            )

    def _generate(
        self,
        prompt: str,
        aspect_ratio: str = "1:1",
        image_size: str = "1K",
        style_notes: str = "",
        **_extra,
    ) -> Dict[str, Any]:
        """Generate an image from a text prompt via a new Chat session.

        Creates a Chat, sends the prompt (with optional style_notes baked in),
        saves the image and Chat history for future refinements.
        """
        if not prompt:
            raise ValueError("'prompt' is required for generate operation")

        self._ensure_generations_table()

        # Compose prompt with style notes
        full_prompt = prompt
        if style_notes:
            full_prompt = f"{prompt}\n\nStyle: {style_notes}"

        # Create Chat session and generate with per-message image config
        chat = self._create_chat()
        msg_config = self._message_config(aspect_ratio=aspect_ratio, image_size=image_size)
        response = chat.send_message(full_prompt, config=msg_config)

        image_bytes, mime_type, response_text = self._extract_image_from_response(response)

        # Save image file
        file_id = self._save_image(image_bytes, mime_type)

        # Create generation record
        session_id = f"sess_{uuid4().hex[:12]}"
        image_id = f"imggen_{uuid4().hex[:12]}"

        self.db.insert('image_generations', {
            'id': image_id,
            'session_id': session_id,
            'file_id': file_id,
            'encrypted__prompt': full_prompt,
            'aspect_ratio': aspect_ratio,
            'created_at': utc_now().isoformat(),
        })

        # Save Chat history for future refinements
        self._save_session(session_id, chat)

        return self._build_content_blocks(
            image_bytes, mime_type,
            {"image_id": image_id, "status": "generated", "prompt": prompt},
        )

    def _refine(
        self,
        image_id: str,
        instructions: str,
        preserve: List[str] | None = None,
        aspect_ratio: str = "",
        image_size: str = "1K",
        **_extra,
    ) -> Dict[str, Any]:
        """Refine an existing image with full lineage context.

        Loads the Chat history from the image's session, recreates the Chat,
        and sends refinement instructions. The SDK replays all prior turns
        (text + images) automatically. Aspect ratio and resolution can be
        changed per-refinement via per-message config.
        """
        from google.genai import types

        if not image_id:
            raise ValueError("'image_id' is required for refine operation")
        if not instructions:
            raise ValueError("'instructions' is required for refine operation")

        # Look up the generation to find its session
        gen_record = self._lookup_generation(image_id)
        session_id = gen_record['session_id']
        # Use provided aspect_ratio, or fall back to the original generation's ratio
        effective_ratio = aspect_ratio or gen_record.get('aspect_ratio', '1:1')

        # Load and restore Chat history
        history_data = self._load_session_history(session_id)
        restored_history = [types.Content.model_validate(item) for item in history_data]
        chat = self._create_chat(history=restored_history)

        # Compose refinement message
        message = instructions
        if preserve:
            preserve_list = ", ".join(preserve)
            message = f"{instructions}\n\nPreserve these elements unchanged: {preserve_list}"

        msg_config = self._message_config(aspect_ratio=effective_ratio, image_size=image_size)
        response = chat.send_message(message, config=msg_config)

        image_bytes, mime_type, response_text = self._extract_image_from_response(response)

        # Save new image
        file_id = self._save_image(image_bytes, mime_type)

        # Create generation record (same session, new image)
        new_image_id = f"imggen_{uuid4().hex[:12]}"
        self.db.insert('image_generations', {
            'id': new_image_id,
            'session_id': session_id,
            'file_id': file_id,
            'encrypted__prompt': instructions,
            'aspect_ratio': effective_ratio,
            'created_at': utc_now().isoformat(),
        })

        # Save updated Chat history
        self._save_session(session_id, chat)

        return self._build_content_blocks(
            image_bytes, mime_type,
            {"image_id": new_image_id, "status": "refined", "previous_image_id": image_id,
             "instructions": instructions},
        )

    def _publish(
        self,
        image_id: str,
        caption: str = "Generated image",
        **_extra,
    ) -> Dict[str, Any]:
        """Publish an image to the user.

        Returns _image_artifact which the orchestrator converts to an inline
        markdown image tag (![caption](/v0/api/images/{file_id})).
        """
        if not image_id:
            raise ValueError("'image_id' is required for publish operation")

        gen_record = self._lookup_generation(image_id)
        file_id = gen_record['file_id']

        # Verify the image file exists
        self._load_image(file_id)

        return {
            "status": "published",
            "image_id": image_id,
            "file_id": file_id,
            "_image_artifact": {
                "file_id": file_id,
                "alt_text": caption,
            },
        }
