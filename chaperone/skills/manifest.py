"""Tool manifests — the typed template for teaching Chaperone to run a tool.

A manifest is a YAML file describing one runnable tool (AlphaFold, LigandMPNN, ...):
what inputs it takes, how to invoke it on this cluster, what resources it needs, and
where its outputs land. The schema below validates those files and turns each tool's
``inputs`` into a Pydantic model so user-supplied parameters are type-checked and
defaulted before they ever reach a job script.

This module deliberately ships NO real tool catalog — only the schema and a
copy-me ``TEMPLATE.yaml``. Add a tool by writing a manifest, not code.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, create_model


class ParamType(str, Enum):
    str = "str"
    int = "int"
    float = "float"
    bool = "bool"
    path = "path"  # a filesystem path, carried as a string


_PYTYPE: dict[ParamType, type] = {
    ParamType.str: str,
    ParamType.int: int,
    ParamType.float: float,
    ParamType.bool: bool,
    ParamType.path: str,
}


class ParamSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: ParamType = ParamType.str
    required: bool = False
    default: Any | None = None
    help: str | None = None


class ResourceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    partition: str = "gpu"
    gres: str | None = "gpu:1"  # None → no --gres line
    cpus: int = 8
    mem: str = "32G"
    time: str = "01:00:00"


class EnvSpec(BaseModel):
    """How to make the tool available on a compute node."""

    model_config = ConfigDict(extra="forbid")
    conda_env: str | None = None
    module_load: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    workdir: str | None = None


class ToolManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    version: str = "unknown"
    inputs: dict[str, ParamSpec] = Field(default_factory=dict)
    env: EnvSpec = Field(default_factory=EnvSpec)
    resources: ResourceSpec = Field(default_factory=ResourceSpec)
    # Shell command template; may reference {input_name} and {out_dir}.
    command: str
    # Named output globs (relative or absolute; may reference {out_dir}).
    outputs: dict[str, str] = Field(default_factory=dict)

    def input_model(self) -> type[BaseModel]:
        """Build a Pydantic model from ``inputs`` for validating user params."""
        fields: dict[str, tuple[Any, Any]] = {}
        for name, spec in self.inputs.items():
            py = _PYTYPE[spec.type]
            if spec.required and spec.default is None:
                fields[name] = (py, ...)
            else:
                fields[name] = (py | None, spec.default)
        return create_model(
            f"{self.name}_Inputs",
            __config__=ConfigDict(extra="forbid"),
            **fields,
        )

    def coerce_params(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Validate + type-coerce ``raw`` params, fill defaults, reject unknown keys.

        Raises ``pydantic.ValidationError`` on missing-required / wrong-type / unknown.
        """
        return self.input_model()(**raw).model_dump()
