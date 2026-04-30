"""Shared configuration for Screen2AX zero-shot VLM evaluation pipeline."""

# vLLM server
# VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_BASE_URL = "http://192.168.20.10:8000/v1"
# MODEL_NAME = "Qwen/Qwen3-VL-30B-A3B-Instruct"
MODEL_NAME = "Qwen/Qwen3-VL-235B-A22B-Instruct"

# Paths
DATASET_DIR = "./data/screen2ax_test"
IMAGES_DIR = f"{DATASET_DIR}/images"
ANNOTATIONS_DIR = f"{DATASET_DIR}/annotations"
RESULTS_DIR = "./results"

# Image resolution for coordinate normalization
NORMALIZE_RANGE = 1000  # normalize coords to 0-1000

# Inference
DEFAULT_MAX_TOKENS = 8192
DEFAULT_TEMPERATURE = 0.0

# Valid AX roles (from Screen2AX paper Appendix A.1)
VALID_ROLES = [
    # Containers
    "AXGroup", "AXOpaqueProviderGroup", "AXRadioGroup", "AXSplitGroup",
    "AXTabGroup", "AXToolbar", "AXWebArea", "AXOutline", "AXBrowser",
    "AXPopover", "AXGrid", "AXList", "AXTable", "AXScrollArea",
    "AXWindow", "AXPage",
    # Text
    "AXStaticText", "AXHeading", "AXLink", "AXListMarker",
    # Controls
    "AXCheckBox", "AXRadioButton", "AXSlider", "AXComboBox",
    "AXScrollBar", "AXButton", "AXPopUpButton", "AXMenuButton",
    "AXDisclosureTriangle", "AXIncrementor", "AXColorWell",
    "AXIncrementorArrow", "AXTextField", "AXTextArea", "AXDateTimeArea",
    # Menus
    "AXMenuItem", "AXMenu", "AXMenuBar",
    # Display
    "AXImage", "AXBusyIndicator", "AXProgressIndicator",
    "AXValueIndicator", "AXRuler",
    # Layout
    "AXSplitter", "AXSheet", "AXGrowArea", "AXCell",
    # Other
    "AXUnknown", "AXGenericElement", "SWTComposite", "JavaAxIgnore",
]

# Set of valid role names (without AX prefix) for quick lookup
VALID_ROLES_NO_PREFIX = {r[2:] if r.startswith("AX") else r for r in VALID_ROLES}

# Role categories for visualization filtering
ROLE_CATEGORIES = {
    "Containers": [
        "AXGroup", "AXOpaqueProviderGroup", "AXRadioGroup", "AXSplitGroup",
        "AXTabGroup", "AXToolbar", "AXWebArea", "AXOutline", "AXBrowser",
        "AXPopover", "AXGrid", "AXList", "AXTable", "AXScrollArea",
        "AXWindow", "AXPage",
    ],
    "Text": ["AXStaticText", "AXHeading", "AXLink", "AXListMarker"],
    "Controls": [
        "AXCheckBox", "AXRadioButton", "AXSlider", "AXComboBox",
        "AXScrollBar", "AXButton", "AXPopUpButton", "AXMenuButton",
        "AXDisclosureTriangle", "AXIncrementor", "AXColorWell",
        "AXIncrementorArrow", "AXTextField", "AXTextArea", "AXDateTimeArea",
    ],
    "Menus": ["AXMenuItem", "AXMenu", "AXMenuBar"],
    "Display": [
        "AXImage", "AXBusyIndicator", "AXProgressIndicator",
        "AXValueIndicator", "AXRuler",
    ],
    "Layout": ["AXSplitter", "AXSheet", "AXGrowArea", "AXCell"],
    "Other": ["AXUnknown", "AXGenericElement", "SWTComposite", "JavaAxIgnore"],
}

# Reverse lookup: role -> category
ROLE_TO_CATEGORY = {}
for _cat, _roles in ROLE_CATEGORIES.items():
    for _role in _roles:
        ROLE_TO_CATEGORY[_role] = _cat

# Color scheme for role categories (R, G, B, A)
CATEGORY_COLORS = {
    "Containers": (66, 133, 244, 80),
    "Text": (52, 168, 83, 80),
    "Controls": (251, 188, 4, 80),
    "Menus": (234, 67, 53, 80),
    "Display": (156, 39, 176, 80),
    "Layout": (158, 158, 158, 80),
    "Other": (121, 85, 72, 80),
}

# Border colors (fully opaque)
CATEGORY_BORDER_COLORS = {
    cat: (r, g, b, 255) for cat, (r, g, b, _) in CATEGORY_COLORS.items()
}

# System prompt for zero-shot evaluation
SYSTEM_PROMPT = """You are an accessibility metadata generator for macOS desktop applications.

Given a screenshot of a macOS application window, generate its complete accessibility tree (AX tree) in the linearized format described below.

## Output Format

Each line represents one UI element. Use 2-space indentation to encode the parent-child hierarchy. The format for each line is:

Role(subrole) [x1,y1,x2,y2] key1="value1" key2="value2"

Where:
- Role: one of the accessibility roles listed below
- (subrole): optional subrole in parentheses (e.g., "close button", "standard window", "switch", "text")
- [x1,y1,x2,y2]: bounding box coordinates normalized to 0–1000 range, where (x1,y1) is top-left and (x2,y2) is bottom-right
- Attributes (include only when present):
  - name="..." — the accessibility name/identifier
  - desc="..." — the accessibility description
  - value="..." — the current value (for inputs, toggles, text fields)

## Valid Roles

Containers: AXGroup, AXOpaqueProviderGroup, AXRadioGroup, AXSplitGroup, AXTabGroup, AXToolbar, AXWebArea, AXOutline, AXBrowser, AXPopover, AXGrid, AXList, AXTable, AXScrollArea, AXWindow, AXPage
Text: AXStaticText, AXHeading, AXLink, AXListMarker
Controls: AXCheckBox, AXRadioButton, AXSlider, AXComboBox, AXScrollBar, AXButton, AXPopUpButton, AXMenuButton, AXDisclosureTriangle, AXIncrementor, AXColorWell, AXIncrementorArrow, AXTextField, AXTextArea, AXDateTimeArea
Menus: AXMenuItem, AXMenu, AXMenuBar
Display: AXImage, AXBusyIndicator, AXProgressIndicator, AXValueIndicator, AXRuler
Layout: AXSplitter, AXSheet, AXGrowArea, AXCell
Other: AXUnknown, AXGenericElement, SWTComposite, JavaAxIgnore

## Rules
1. Indent child elements by 2 spaces relative to their parent.
2. The root element should be the application window with no indentation.
3. Group related UI elements under container nodes (AXGroup, AXScrollArea, AXToolbar, etc.).
4. Every visible interactive or informational element must appear in the tree.
5. Use ONLY roles from the valid roles list above.
6. Output ONLY the tree — no explanations, no markdown code blocks, no commentary."""

USER_PROMPT = "Generate the complete accessibility tree for this screenshot."

# ---------------------------------------------------------------------------
# Simplified 7-class role mapping (Screen2AX paper)
# ---------------------------------------------------------------------------

# Maps every original AX role to one of 7 classes, or None to drop.
SIMPLIFIED_ROLE_MAPPING = {
    # AXButton — all interactable button-like elements
    "AXButton": "AXButton",
    "AXCheckBox": "AXButton",
    "AXRadioButton": "AXButton",
    "AXPopUpButton": "AXButton",
    "AXMenuButton": "AXButton",
    "AXIncrementor": "AXButton",
    "AXIncrementorArrow": "AXButton",
    "AXColorWell": "AXButton",
    "AXSlider": "AXButton",
    "AXScrollBar": "AXButton",
    "AXMenuItem": "AXButton",
    "AXValueIndicator": "AXButton",
    # AXDisclosureTriangle — kept separate
    "AXDisclosureTriangle": "AXDisclosureTriangle",
    # AXLink — clickable text elements
    "AXLink": "AXLink",
    # AXTextArea — all input-related elements
    "AXTextArea": "AXTextArea",
    "AXTextField": "AXTextArea",
    "AXComboBox": "AXTextArea",
    "AXDateTimeArea": "AXTextArea",
    # AXImage — decorative/functional images and indicators
    "AXImage": "AXImage",
    "AXBusyIndicator": "AXImage",
    "AXProgressIndicator": "AXImage",
    "AXRuler": "AXImage",
    # AXStaticText — all text including headings
    "AXStaticText": "AXStaticText",
    "AXHeading": "AXStaticText",
    "AXListMarker": "AXStaticText",
    # AXGroup — all containers, structural groupings, menus, layout
    "AXGroup": "AXGroup",
    "AXOpaqueProviderGroup": "AXGroup",
    "AXRadioGroup": "AXGroup",
    "AXSplitGroup": "AXGroup",
    "AXTabGroup": "AXGroup",
    "AXToolbar": "AXGroup",
    "AXWebArea": "AXGroup",
    "AXOutline": "AXGroup",
    "AXBrowser": "AXGroup",
    "AXPopover": "AXGroup",
    "AXGrid": "AXGroup",
    "AXList": "AXGroup",
    "AXTable": "AXGroup",
    "AXScrollArea": "AXGroup",
    "AXWindow": "AXGroup",
    "AXPage": "AXGroup",
    "AXMenu": "AXGroup",
    "AXMenuBar": "AXGroup",
    "AXSheet": "AXGroup",
    "AXSplitter": "AXGroup",
    "AXGrowArea": "AXGroup",
    "AXCell": "AXGroup",
    # Dropped (rare / unmapped)
    "AXUnknown": None,
    "AXGenericElement": None,
    "SWTComposite": None,
    "JavaAxIgnore": None,
}

# The 7 target classes
SIMPLIFIED_ROLES = sorted({v for v in SIMPLIFIED_ROLE_MAPPING.values() if v is not None})

# Categories for visualization (each class is its own category)
SIMPLIFIED_ROLE_CATEGORIES = {
    "Button": ["AXButton"],
    "DisclosureTriangle": ["AXDisclosureTriangle"],
    "Link": ["AXLink"],
    "TextArea": ["AXTextArea"],
    "Image": ["AXImage"],
    "StaticText": ["AXStaticText"],
    "Group": ["AXGroup"],
}

SIMPLIFIED_ROLE_TO_CATEGORY = {}
for _cat, _roles in SIMPLIFIED_ROLE_CATEGORIES.items():
    for _role in _roles:
        SIMPLIFIED_ROLE_TO_CATEGORY[_role] = _cat

SIMPLIFIED_CATEGORY_COLORS = {
    "Button": (251, 188, 4, 80),
    "DisclosureTriangle": (255, 112, 67, 80),
    "Link": (41, 182, 246, 80),
    "TextArea": (171, 71, 188, 80),
    "Image": (156, 39, 176, 80),
    "StaticText": (52, 168, 83, 80),
    "Group": (66, 133, 244, 80),
}

SIMPLIFIED_CATEGORY_BORDER_COLORS = {
    cat: (r, g, b, 255) for cat, (r, g, b, _) in SIMPLIFIED_CATEGORY_COLORS.items()
}

# Simplified system prompt (7 roles only)
# SIMPLIFIED_SYSTEM_PROMPT = """You are an accessibility metadata generator for macOS desktop applications.

# Given a screenshot of a macOS application window, generate its complete accessibility tree (AX tree) in the linearized format described below.

# ## Output Format

# Each line represents one UI element. Use 2-space indentation to encode the parent-child hierarchy. The format for each line is:

# Role(subrole) [x1,y1,x2,y2] key1="value1" key2="value2"

# Where:
# - Role: one of the accessibility roles listed below
# - (subrole): optional subrole in parentheses (e.g., "close button", "standard window", "switch", "text")
# - [x1,y1,x2,y2]: bounding box coordinates normalized to 0–1000 range, where (x1,y1) is top-left and (x2,y2) is bottom-right
# - Attributes (include only when present):
#   - name="..." — the accessibility name/identifier
#   - desc="..." — the accessibility description
#   - value="..." — the current value (for inputs, toggles, text fields)

# ## Valid Roles

# Containers: AXGroup
# Interactive: AXButton, AXDisclosureTriangle, AXLink
# Input: AXTextArea
# Display: AXImage, AXStaticText

# ## Rules
# 1. Indent child elements by 2 spaces relative to their parent.
# 2. The root element should be the application window with no indentation.
# 3. Group related UI elements under AXGroup container nodes.
# 4. Every visible interactive or informational element must appear in the tree.
# 5. Use ONLY roles from the valid roles list above.
# 6. Output ONLY the tree — no explanations, no markdown code blocks, no commentary."""

SIMPLIFIED_SYSTEM_PROMPT = """You are an accessibility metadata generator for macOS desktop applications.

Given a screenshot of a macOS application window, generate its complete accessibility tree (AX tree) in the linearized format described below.

## Output Format

Each line represents one UI element. Use 2-space indentation to encode the parent-child hierarchy. The format for each line is:

Role(subrole) [x1,y1,x2,y2] key1="value1" key2="value2"

Where:
- Role: one of the accessibility roles listed below
- (subrole): optional subrole in parentheses (e.g., "close button", "standard window", "switch", "text")
- [x1,y1,x2,y2]: bounding box coordinates normalized to 0–1000 range, where (x1,y1) is top-left and (x2,y2) is bottom-right
- Attributes (include only when present):
  - name="..." — the accessibility name/identifier
  - desc="..." — the accessibility description
  - value="..." — the current value (for inputs, toggles, text fields)

## Valid Roles

AXGroup — Structural containers that group related elements: window frames, toolbars, sidebars, menu bars, scroll areas, tab groups, table rows, form sections, or any logical grouping of child elements. Use AXGroup to represent the hierarchical structure of the interface.

AXButton — All interactive controls that trigger an action when clicked: buttons, checkboxes, radio buttons, toggle switches, pop-up buttons, menu items, steppers, sliders, scroll bar handles, and color wells.

AXDisclosureTriangle — A specific control that expands or collapses a section of content. Visually appears as a small triangle or chevron (▶/▼) next to expandable content.

AXLink — Clickable text that navigates to another location or triggers an action. Visually distinguished from regular text by color, underline, or cursor change on hover.

AXTextArea — All editable input fields: single-line text fields, multi-line text areas, search boxes, combo boxes, and date/time pickers.

AXImage — Visual non-text content: icons, photos, illustrations, decorative images, progress indicators, loading spinners, and visual rulers.

AXStaticText — All non-editable, non-interactive text displayed in the interface: labels, headings, descriptions, status messages, list markers, and any read-only text content.

## Rules
1. Indent child elements by 2 spaces relative to their parent.
2. The root element should have no indentation.
3. Group related UI elements under AXGroup container nodes to reflect the visual layout hierarchy.
4. Every visible interactive or informational element must appear in the tree exactly once.
5. Do NOT repeat elements. Each UI element in the screenshot corresponds to exactly one line in the output.
6. Use ONLY roles from the valid roles list above.
7. Output ONLY the tree — no explanations, no markdown code blocks, no commentary before or after the tree."""
