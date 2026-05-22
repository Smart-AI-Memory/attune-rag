"""Shared compiled regex patterns used across the editor subpackage.

These three patterns were copy-pasted across ``lint.py``, ``references.py``,
and ``rename.py`` with identical sources; centralising them here avoids the
risk of one file's copy drifting away from the others when the markdown
dialect evolves.
"""

from __future__ import annotations

import re

# Code-fence opener (matches both ``` and ~~~ at start of line).
FENCE_RE = re.compile(r"^(```|~~~)")

# YAML top-level key at the start of a frontmatter line.
TOP_LEVEL_KEY_RE = re.compile(r"^([A-Za-z_][\w-]*)\s*:")

# Wiki-style alias reference, e.g. [[some-alias]]. The negative
# lookbehind excludes escaped \[[…]].
ALIAS_REF_RE = re.compile(r"(?<!\\)\[\[([^\[\]\n]+?)\]\]")
