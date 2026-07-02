"""
Maps task names to model tiers. One place to change routing for the whole app.
"""
from app.services.llm_client import ModelTier

_ROUTING_TABLE: dict[str, str] = {
    "chunk_summarise":  ModelTier.FAST,
    "docstring_draft":  ModelTier.FAST,
    "file_classify":    ModelTier.FAST,
    "doc_assemble":     ModelTier.REASONING,
    "qa_answer":        ModelTier.REASONING,
    "pr_description":   ModelTier.REASONING,
    "readme_generate":  ModelTier.REASONING,
    "guide_generate":   ModelTier.REASONING,
}


def route_task(task: str) -> str:
    tier = _ROUTING_TABLE.get(task)
    if tier is None:
        raise ValueError(f"Unknown task '{task}'. Valid: {sorted(_ROUTING_TABLE.keys())}")
    return tier
