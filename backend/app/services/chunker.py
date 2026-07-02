"""
Splits a file's content into Chunk objects.

Two strategies, one shared output type:
  - AST chunking (tree-sitter): splits Python/JS/TS at function and class
    boundaries, so an LLM summarising a chunk sees a whole function, not
    half of one.
  - Fallback chunking: fixed-size line windows with overlap, used for every
    other supported extension (markdown, json, yaml, txt) and as a safety
    net if AST parsing fails for any reason.

Callers (the ingestion service) only ever call chunk_file() and only ever
receive Chunk objects. They never need to know which strategy ran.
"""
import hashlib
from dataclasses import dataclass, field

from app.core.config import settings
from app.core.logging import get_logger
from app.services.walker import DiscoveredFile

logger = get_logger(__name__)

try:
    from tree_sitter_languages import get_parser
    _TREE_SITTER_AVAILABLE = True
except ImportError:  
    _TREE_SITTER_AVAILABLE = False

# Node types that count as a "chunkable unit" per language.
AST_CHUNK_NODE_TYPES: dict[str, set[str]] = {
    "python": {"function_definition", "class_definition"},
    "javascript": {
        "function_declaration", "class_declaration",
        "method_definition", "arrow_function",
    },
    "typescript": {
        "function_declaration", "class_declaration",
        "method_definition", "arrow_function", "interface_declaration",
    },
}

TREE_SITTER_GRAMMAR_NAME = {
    "python": "python",
    "javascript": "javascript",
    "typescript": "typescript",
}


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    file_path: str
    language: str
    chunk_type: str  # "function" | "class" | "block" | "document"
    symbol_name: str | None
    start_line: int
    end_line: int
    text: str
    metadata: dict = field(default_factory=dict)


def _make_chunk_id(file_path: str, start_line: int, text: str) -> str:
    """Deterministic ID from file path + position + content hash.

    Deterministic IDs mean re-chunking an unchanged file produces the same
    chunk_id, which is what lets the embedding step skip work it already did.
    """
    digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"{file_path}:{start_line}:{digest}"


def _extract_symbol_name(node, source_bytes: bytes) -> str | None:
    """Pulls the identifier name out of a function/class node, if present."""
    for child in node.children:
        if child.type in ("identifier", "property_identifier", "type_identifier"):
            return source_bytes[child.start_byte:child.end_byte].decode(
                "utf-8", errors="ignore"
            )
    return None


def _chunk_with_ast(file: DiscoveredFile, content: str) -> list[Chunk] | None:
    """
    Returns AST-based chunks, or None if this language/file can't be parsed,
    so the caller knows to fall back rather than silently returning [].
    """
    if not _TREE_SITTER_AVAILABLE:
        return None

    grammar_name = TREE_SITTER_GRAMMAR_NAME.get(file.language)
    chunk_node_types = AST_CHUNK_NODE_TYPES.get(file.language)
    if grammar_name is None or chunk_node_types is None:
        return None

    try:
        parser = get_parser(grammar_name)
        source_bytes = content.encode("utf-8", errors="ignore")
        tree = parser.parse(source_bytes)
    except Exception as exc:  # tree-sitter raises various low-level errors
        logger.warning("tree-sitter failed to parse %s: %s", file.relative_path, exc)
        return None

    chunks: list[Chunk] = []

    def visit(node):
        if node.type in chunk_node_types:
            text = source_bytes[node.start_byte:node.end_byte].decode(
                "utf-8", errors="ignore"
            )
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            chunk_type = "class" if "class" in node.type else "function"
            chunks.append(
                Chunk(
                    chunk_id=_make_chunk_id(file.relative_path, start_line, text),
                    file_path=file.relative_path,
                    language=file.language,
                    chunk_type=chunk_type,
                    symbol_name=_extract_symbol_name(node, source_bytes),
                    start_line=start_line,
                    end_line=end_line,
                    text=text,
                )
            )
            return  # don't descend into nested functions/classes separately

        for child in node.children:
            visit(child)

    visit(tree.root_node)

    if not chunks:
        return None 

    return chunks


def _chunk_with_fixed_windows(file: DiscoveredFile, content: str) -> list[Chunk]:
    """
    Fallback strategy: fixed-size line windows with overlap.

    Used for markdown/json/yaml/txt, and for any code file where AST
    chunking returned nothing, so every supported file always produces
    at least one chunk.
    """
    lines = content.splitlines()
    if not lines:
        return []

    window = settings.fallback_chunk_lines
    overlap = settings.fallback_chunk_overlap
    step = max(window - overlap, 1)

    chunks: list[Chunk] = []
    for start in range(0, len(lines), step):
        end = min(start + window, len(lines))
        text = "\n".join(lines[start:end])
        if not text.strip():
            continue
        chunks.append(
            Chunk(
                chunk_id=_make_chunk_id(file.relative_path, start + 1, text),
                file_path=file.relative_path,
                language=file.language,
                chunk_type="document" if file.language in ("markdown", "text") else "block",
                symbol_name=None,
                start_line=start + 1,
                end_line=end,
                text=text,
            )
        )
        if end >= len(lines):
            break

    return chunks


def chunk_file(file: DiscoveredFile) -> list[Chunk]:
    """
    Single entry point: reads the file and returns its chunks, trying AST
    chunking first for supported languages and falling back to fixed-size
    windows for everything else or on any parse failure.
    """
    try:
        content = file.absolute_path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        logger.warning("Could not read %s: %s", file.relative_path, exc)
        return []

    if file.language in AST_CHUNK_NODE_TYPES:
        ast_chunks = _chunk_with_ast(file, content)
        if ast_chunks is not None:
            return ast_chunks

    return _chunk_with_fixed_windows(file, content)
