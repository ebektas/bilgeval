import hashlib
from typing import Any

from .manifest import stable_json_sha256
from .registry_bundle import ModelProfile, ModelSerializationSpec, PromptRendererSpec


def renderer_manifest(renderer: PromptRendererSpec) -> dict[str, Any]:
    template_hash = (
        "sha256:" + hashlib.sha256(renderer.template.encode("utf-8")).hexdigest()
        if renderer.mode == "template" and renderer.template is not None
        else None
    )
    return {
        "mode": renderer.mode,
        "template_id": renderer.template_id,
        "template_version": renderer.template_version,
        "template_hash": template_hash,
    }


def serialization_manifest(
    serialization: ModelSerializationSpec,
    *,
    generation_api: str,
) -> dict[str, Any]:
    value = {
        "tokenizer": {
            "id": serialization.tokenizer_id,
            "revision": serialization.tokenizer_revision,
        },
        "logprob": renderer_manifest(serialization.logprob),
        "generation": {
            **renderer_manifest(serialization.generation),
            "endpoint_template_application": (
                "server" if generation_api == "chat" else "none"
            ),
        },
    }
    value["fingerprint"] = stable_json_sha256(value)
    return value


def profile_serialization(profile: ModelProfile | None) -> ModelSerializationSpec:
    return profile.serialization if profile is not None else ModelSerializationSpec()


def render_model_prompt(prompt: str, renderer: PromptRendererSpec | None) -> str:
    if renderer is None or renderer.mode in {"plain", "server_chat"}:
        return prompt
    if renderer.template is None:
        raise ValueError("A template renderer requires an inline template.")
    return renderer.template.format(prompt=prompt)
