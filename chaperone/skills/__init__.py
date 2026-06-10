"""Tool manifests: the typed template for adding runnable tools."""

from chaperone.skills.manifest import ParamSpec, ParamType, ToolManifest
from chaperone.skills.registry import SkillRegistry
from chaperone.skills.render import RenderedJob, render_job

__all__ = [
    "ParamSpec",
    "ParamType",
    "RenderedJob",
    "SkillRegistry",
    "ToolManifest",
    "render_job",
]
