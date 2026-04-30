"""Parse the linearized AX tree format into structured nodes."""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AXNode:
    role: str
    subrole: Optional[str] = None
    bbox: Optional[List[int]] = None  # [x1, y1, x2, y2] in 0-1000 range
    name: Optional[str] = None
    desc: Optional[str] = None
    value: Optional[str] = None
    depth: int = 0
    children: List["AXNode"] = field(default_factory=list)
    node_id: int = -1


# Line regex pattern:
# Role(subrole) [x1,y1,x2,y2] name="..." desc="..." value="..."
# Role [x1,y1,x2,y2] desc="..."
# etc. All parts except Role are optional.
LINE_PATTERN = re.compile(
    r"^(?P<role>\w+)"  # Role name
    r"(?:\((?P<subrole>[^)]+)\))?"  # Optional (subrole)
    r"(?:\s*\[(?P<bbox>[^\]]+)\])?"  # Optional [x1,y1,x2,y2]
    r"(?P<attrs>.*)"  # Remaining attributes
)

ATTR_PATTERN = re.compile(r'(\w+)="([^"]*)"')


def parse_line(line: str) -> Optional[AXNode]:
    """Parse a single line into an AXNode. Returns None if unparseable."""
    stripped = line.lstrip()
    if not stripped:
        return None

    # Skip lines that are clearly not AX tree lines (e.g. "..." truncation)
    if stripped.startswith("...") or stripped.startswith("#"):
        return None

    depth = (len(line) - len(stripped)) // 2

    match = LINE_PATTERN.match(stripped)
    if not match:
        logger.warning(f"Could not parse line: {stripped!r}")
        return None

    role = match.group("role")
    subrole = match.group("subrole")

    bbox = None
    bbox_str = match.group("bbox")
    if bbox_str:
        try:
            coords = [x.strip() for x in bbox_str.split(",")]
            # Filter out empty strings from trailing commas
            coords = [x for x in coords if x]
            bbox = [int(float(x)) for x in coords]
            if len(bbox) != 4:
                logger.warning(f"Bbox has {len(bbox)} coords (expected 4): {bbox_str!r}")
                bbox = None
        except ValueError:
            logger.warning(f"Could not parse bbox: {bbox_str!r}")
            bbox = None

    attrs_str = match.group("attrs")
    attrs = dict(ATTR_PATTERN.findall(attrs_str))

    return AXNode(
        role=role,
        subrole=subrole,
        bbox=bbox,
        name=attrs.get("name"),
        desc=attrs.get("desc"),
        value=attrs.get("value"),
        depth=depth,
    )


def parse_tree(
    text: str,
    role_mapping: Optional[Dict[str, Optional[str]]] = None,
) -> List[AXNode]:
    """Parse full linearized AX tree text into a list of AXNodes with hierarchy.

    Returns a flat list of all nodes. Parent-child relationships are encoded
    via each node's `children` list (built from indentation).

    If *role_mapping* is provided, each node's role is remapped:
    - Mapped to a string → role is replaced.
    - Mapped to ``None`` → node is dropped.
    - Role not in mapping → node is dropped.
    """
    nodes: List[AXNode] = []
    stack: list[tuple[int, AXNode]] = []

    for line in text.split("\n"):
        if not line.strip():
            continue

        node = parse_line(line)
        if node is None:
            continue

        # Apply role mapping if provided
        if role_mapping is not None:
            mapped = role_mapping.get(node.role)
            if mapped is None:
                continue
            node.role = mapped

        node.node_id = len(nodes)

        # Build hierarchy via indentation stack
        while stack and stack[-1][0] >= node.depth:
            stack.pop()
        if stack:
            stack[-1][1].children.append(node)

        stack.append((node.depth, node))
        nodes.append(node)

    return nodes


def get_root_nodes(nodes: List[AXNode]) -> List[AXNode]:
    """Return only the root-level nodes (depth 0) from a parsed node list."""
    return [n for n in nodes if n.depth == 0]


def tree_to_text(nodes: List[AXNode]) -> str:
    """Convert list of AXNodes back to linearized text (for display)."""
    lines = []
    for node in nodes:
        indent = "  " * node.depth
        parts = [node.role]
        if node.subrole:
            parts[0] += f"({node.subrole})"
        if node.bbox:
            parts.append(f"[{','.join(str(x) for x in node.bbox)}]")
        if node.name:
            parts.append(f'name="{node.name}"')
        if node.desc:
            parts.append(f'desc="{node.desc}"')
        if node.value:
            parts.append(f'value="{node.value}"')
        lines.append(indent + " ".join(parts))
    return "\n".join(lines)


def is_leaf(node: AXNode) -> bool:
    """Check if a node is a leaf (has no children)."""
    return len(node.children) == 0
