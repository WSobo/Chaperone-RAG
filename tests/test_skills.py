"""Tool-manifest validation, param coercion, and rendering (pydantic/yaml only)."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from chaperone.skills.manifest import ToolManifest
from chaperone.skills.registry import SkillRegistry
from chaperone.skills.render import render_job

_CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "skills"


def _manifest() -> ToolManifest:
    return ToolManifest(
        name="demo",
        command="echo {message} > {out_dir}/r.txt",
        inputs={
            "message": {"type": "str", "required": True},
            "n": {"type": "int", "default": 3},
        },
        outputs={"r": "{out_dir}/r.txt"},
        resources={"gres": None, "partition": "debug"},
    )


def test_coerce_fills_defaults_and_coerces_types():
    assert _manifest().coerce_params({"message": "hi", "n": "5"}) == {"message": "hi", "n": 5}


def test_missing_required_raises():
    with pytest.raises(ValidationError):
        _manifest().coerce_params({"n": 1})


def test_unknown_param_raises():
    with pytest.raises(ValidationError):
        _manifest().coerce_params({"message": "hi", "bogus": 1})


def test_render_substitutes_params_and_omits_null_gres():
    job = render_job(_manifest(), {"message": "hello"}, out_dir="/tmp/o", job_name="demo-1")
    assert "echo hello > /tmp/o/r.txt" in job.script
    assert "--gres" not in job.script  # gres is null
    assert "#SBATCH --partition=debug" in job.script
    assert job.params["n"] == 3  # default applied
    assert job.output_globs == {"r": "{out_dir}/r.txt"}  # globs stay templated


def test_registry_loads_example_and_ignores_template():
    names = SkillRegistry(_CONFIGS).names()
    assert "echo-demo" in names
    assert "my-tool" not in names  # TEMPLATE.yaml is ignored, not loaded as a tool
