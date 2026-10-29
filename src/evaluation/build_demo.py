"""
Phase 7 -- assemble the self-contained interactive demo.

Injects the exported replay payload into the page template, producing one HTML
file with no external dependencies beyond a Google Fonts stylesheet (which
degrades to a system stack offline). The file opens by double-clicking it: no
server, no install, nothing to go wrong in front of a review panel.

Usage
-----
    python -m src.evaluation.export_demo_data     # refresh the payload first
    python -m src.evaluation.build_demo
"""

from __future__ import annotations

import sys
from pathlib import Path

from src import config

TEMPLATE = Path(__file__).resolve().parent / "demo_template.html"
DATA = config.PROJECT_ROOT / "reports" / "demo" / "replay_data.json"
OUT = config.PROJECT_ROOT / "reports" / "demo" / "race_engineer_demo.html"
PLACEHOLDER = "/*__DATA__*/"


def build() -> Path:
    for path in (TEMPLATE, DATA):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python -m src.evaluation.export_demo_data` first."
            )

    template = TEMPLATE.read_text(encoding="utf-8")
    if PLACEHOLDER not in template:
        raise ValueError(f"{TEMPLATE.name} is missing the {PLACEHOLDER} placeholder")

    payload = DATA.read_text(encoding="utf-8")
    # The payload is JSON going into a <script>; the only sequence that could
    # close the block early is a literal "</script>", which JSON cannot contain
    # unescaped. Guard anyway rather than assume.
    if "</script" in payload.lower():
        raise ValueError("payload contains a script-closing sequence")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(template.replace(PLACEHOLDER, payload), encoding="utf-8")
    return OUT


def main() -> int:
    out = build()
    print(f"{out}")
    print(f"  {out.stat().st_size / 1024:,.0f} KB, self-contained — open it by double-clicking")
    return 0


if __name__ == "__main__":
    sys.exit(main())
