"""Small, removable compatibility bridge for an upstream Ragas import regression."""

from __future__ import annotations

import sys
from types import ModuleType


def install_legacy_vertexai_import() -> None:
    """Expose LangChain's retired VertexAI chat import only while loading Ragas.

    Ragas 0.3.9--0.4.3 import the retired ``langchain_community`` chat
    module even when a project does not use Vertex AI.  Our evaluator uses
    Gemini through the OpenAI-compatible client, so this placeholder is never
    instantiated; it simply lets Ragas load with LangChain 1.x.
    """
    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return

    module = ModuleType(module_name)

    class ChatVertexAI:  # pragma: no cover - defensive import compatibility only
        pass

    module.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = module
