"""Job lifecycle over the LocalScheduler — runs end-to-end on CPU, no SLURM."""

from pathlib import Path

from chaperone.jobs import JobRunner, JobState, LocalScheduler, RunStore
from chaperone.settings import Settings
from chaperone.skills.manifest import ToolManifest


def _settings(tmp_path) -> Settings:
    s = Settings()
    s.paths.runs_dir = tmp_path / "runs"
    s.jobs.poll_interval = 0.05
    return s


def _echo_manifest() -> ToolManifest:
    return ToolManifest(
        name="echo-demo",
        command='echo "{message}" > {out_dir}/result.txt',
        inputs={"message": {"type": "str", "required": True}},
        outputs={"result": "{out_dir}/result.txt"},
        resources={"gres": None, "partition": "debug"},
        env={},
    )


def test_full_lifecycle_completes_and_collects_outputs(tmp_path):
    settings = _settings(tmp_path)
    runner = JobRunner(settings, LocalScheduler())

    result = runner.run(_echo_manifest(), {"message": "hello world"}, watch=True)
    rec = result.record

    assert result.success
    assert rec.state == JobState.completed
    assert rec.exit_code == 0
    assert rec.outputs["result"], "expected result.txt to be collected"
    assert Path(rec.outputs["result"][0]).read_text().strip() == "hello world"


def test_run_is_persisted_and_listable(tmp_path):
    settings = _settings(tmp_path)
    runner = JobRunner(settings, LocalScheduler())
    rec = runner.run(_echo_manifest(), {"message": "hi"}, watch=True).record

    store = RunStore(settings.paths.runs_dir)
    loaded = store.load(rec.run_id)
    assert loaded.run_id == rec.run_id
    assert loaded.state == JobState.completed
    assert any(r.run_id == rec.run_id for r in store.list())


def test_dry_run_render_writes_nothing(tmp_path):
    settings = _settings(tmp_path)
    runner = JobRunner(settings, LocalScheduler())

    rendered = runner.render(_echo_manifest(), {"message": "hi"})
    assert 'echo "hi"' in rendered.script
    # render() must not create run artifacts
    assert not list(settings.paths.runs_dir.glob("*/job.sh")) if settings.paths.runs_dir.exists() else True
