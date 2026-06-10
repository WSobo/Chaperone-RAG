"""Load tool manifests from a directory of YAML files."""

from __future__ import annotations

from pathlib import Path

import yaml

from chaperone.skills.manifest import ToolManifest

# The copy-me template is not a runnable tool; never load it as one.
_IGNORED_STEMS = {"TEMPLATE"}


class SkillRegistry:
    def __init__(self, skills_dir: Path) -> None:
        self._dir = Path(skills_dir)

    def names(self) -> list[str]:
        return sorted(m.name for m in self.all())

    def all(self) -> list[ToolManifest]:
        if not self._dir.is_dir():
            return []
        out: list[ToolManifest] = []
        for path in sorted(self._dir.glob("*.yaml")):
            if path.stem in _IGNORED_STEMS:
                continue
            out.append(_load(path))
        return out

    def get(self, name: str) -> ToolManifest:
        for manifest in self.all():
            if manifest.name == name:
                return manifest
        available = ", ".join(self.names()) or "(none)"
        raise KeyError(f"Unknown tool '{name}'. Available: {available}")


def _load(path: Path) -> ToolManifest:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return ToolManifest.model_validate(data)
