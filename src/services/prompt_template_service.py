"""AgentCore Platform v1.0"""

# Service layer: Jinja2-based prompt rendering with input sanitization.
# Escapes Jinja2 delimiters in context_vars values to prevent prompt injection.
# No external dependencies beyond Jinja2.
# PB-4: Level-0 SDK (agenticstar) imports are prohibited in this file.

from __future__ import annotations

from typing import Any

from jinja2 import BaseLoader, Environment


class PromptTemplateService:
    """Render Jinja2 prompt templates with sanitized context variables.

    Prompt injection mitigation: all context_vars values have Jinja2
    delimiters ({{ and }}) escaped before rendering, so caller-supplied
    values cannot inject template directives.
    """

    def __init__(self) -> None:
        self._env = Environment(loader=BaseLoader(), autoescape=False)  # noqa: S701

    def render(
        self,
        template_str: str,
        context_vars: dict[str, Any],
        validation_error: str | None = None,
    ) -> str:
        """Render a Jinja2 template with sanitized context variables.

        Args:
            template_str: Jinja2 template string (from user_input or config).
            context_vars: Variables to substitute into the template.
                All string values are sanitized to escape {{ / }} delimiters.
            validation_error: If non-None (retry path), appended as a
                correction instruction at the end of the rendered prompt.

        Returns:
            Rendered prompt string.

        Raises:
            TemplateError: If Jinja2 template rendering fails.
        """
        sanitized = self._sanitize_context_vars(context_vars)
        template = self._env.from_string(template_str)
        rendered = template.render(**sanitized)
        if validation_error:
            rendered += (
                f"\n\n[CORRECTION REQUIRED]\n"
                f"Your previous response did not conform to the required JSON schema.\n"
                f"Validation error: {validation_error}\n"
                f"Please correct your response and ensure it matches the schema exactly."
            )
        return rendered

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_context_vars(context_vars: dict[str, Any]) -> dict[str, Any]:
        """Escape Jinja2 delimiters in all string values of context_vars.

        Replaces {{ with { { and }} with } } so that caller-supplied string
        values cannot inject Jinja2 template directives if the rendered output
        is ever re-processed or logged in a context that evaluates templates.
        """
        sanitized: dict[str, Any] = {}
        for key, value in context_vars.items():
            if isinstance(value, str):
                escaped = value.replace("{{", "{ {").replace("}}", "} }")
                sanitized[key] = escaped
            else:
                sanitized[key] = value
        return sanitized
