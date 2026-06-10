"""Typer CLI — the user-facing entry point (`chaperone ...`)."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from chaperone.schemas import Answer
from chaperone.settings import get_settings

app = typer.Typer(
    add_completion=False,
    help="Chaperone-RAG: cited, grounded RAG for protein design and engineering.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def ingest(
    sources: list[str] = typer.Argument(None, help="Files, directories, or URLs to ingest."),
) -> None:
    """Ingest documents/URLs into the vector store (defaults to the papers dir)."""
    from chaperone.app import build_ingestor

    items = list(sources) if sources else [str(get_settings().paths.papers_dir)]
    n = build_ingestor().ingest_paths(items)
    console.print(f"[green]Indexed {n} chunks[/green] from {len(items)} source(s).")


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to answer."),
    agent: bool = typer.Option(False, "--agent", help="Use the corrective-RAG agent."),
) -> None:
    """Answer a single question with inline citations."""
    engine = _engine(agent)
    _render(engine.invoke(question))


@app.command()
def chat(
    agent: bool = typer.Option(False, "--agent", help="Use the corrective-RAG agent."),
) -> None:
    """Interactive REPL."""
    engine = _engine(agent)
    _welcome()
    while True:
        try:
            question = console.input("\n[green]Ask Chaperone >[/green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            return
        if question.lower() in {"exit", "quit", "q"}:
            return
        if question:
            _render(engine.invoke(question))


@app.command(name="eval")
def evaluate() -> None:
    """Run the evaluation harness against the bundled golden set."""
    from chaperone.eval import run_eval

    console.print("[dim]Running evaluation (builds an ephemeral corpus)…[/dim]")
    report = run_eval()
    console.print(Panel(report.summary(), title="Evaluation", expand=False))


@app.command()
def info() -> None:
    """Print resolved settings (backend, models, retrieval knobs)."""
    s = get_settings()
    table = Table(title="Chaperone settings", show_header=False)
    rows = {
        "llm.backend": s.llm.backend,
        "llm.model_id": s.llm.model_id,
        "embedding.model": s.embedding.model_name,
        "embedding.device": s.embedding.device,
        "retrieval.top_k": s.retrieval.top_k,
        "retrieval.rerank_top_n": s.retrieval.rerank_top_n,
        "retrieval.reranker": s.retrieval.reranker_model if s.retrieval.use_reranker else "off",
        "retrieval.multi_query": s.retrieval.use_multi_query,
        "retrieval.hyde": s.retrieval.use_hyde,
        "paths.vector_db": str(s.paths.vector_db),
    }
    for k, v in rows.items():
        table.add_row(k, str(v))
    console.print(table)


@app.command()
def skills() -> None:
    """List available tool manifests (configs/skills/*.yaml)."""
    from chaperone.app import build_registry

    manifests = build_registry().all()
    if not manifests:
        console.print(
            "[dim]No tool manifests found. Copy configs/skills/TEMPLATE.yaml to add one.[/dim]"
        )
        return
    table = Table(title="Tools")
    table.add_column("name", style="cyan")
    table.add_column("version")
    table.add_column("description")
    for m in manifests:
        table.add_row(m.name, m.version, m.description)
    console.print(table)


@app.command()
def run(
    tool: str = typer.Argument(..., help="Tool name (see `chaperone skills`)."),
    param: list[str] = typer.Option([], "--param", "-p", help="Tool parameter key=value (repeatable)."),
    submit: bool = typer.Option(False, "--submit", help="Queue the job (default: dry-run / render only)."),
    watch: bool = typer.Option(True, "--watch/--no-watch", help="Monitor until the job finishes."),
) -> None:
    """Render — and optionally submit + track — a tool run from its manifest."""
    from pydantic import ValidationError

    from chaperone.app import build_job_runner, build_registry

    settings = get_settings()
    try:
        manifest = build_registry(settings).get(tool)
    except KeyError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e

    runner = build_job_runner(settings)
    try:
        if not submit:
            rendered = runner.render(manifest, _parse_params(param))
            console.print(Panel(rendered.script, title=f"Dry run · {tool} (not submitted)", expand=False))
            console.print("[dim]Add --submit to queue this job.[/dim]")
            return
        result = runner.run(manifest, _parse_params(param), watch=watch)
    except ValidationError as e:
        console.print(f"[red]Invalid parameters for {tool}:[/red]\n{e}")
        raise typer.Exit(1) from e
    _render_run(result.record)


@app.command()
def runs(limit: int = typer.Option(20, help="Show the most recent N runs.")) -> None:
    """List recorded tool runs (provenance)."""
    from chaperone.jobs import RunStore

    records = RunStore(get_settings().paths.runs_dir).list()[:limit]
    if not records:
        console.print("[dim]No runs yet.[/dim]")
        return
    table = Table(title="Runs")
    for col in ("run_id", "tool", "state", "created"):
        table.add_column(col)
    for r in records:
        table.add_row(r.run_id, r.tool, r.state.value, r.created_at)
    console.print(table)


# --- helpers -----------------------------------------------------------------


def _parse_params(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise typer.BadParameter(f"Expected key=value, got: {item}")
        key, value = item.split("=", 1)
        out[key.strip()] = value
    return out


def _render_run(record) -> None:  # noqa: ANN001 - JobRecord
    color = "green" if record.succeeded else "yellow"
    lines = [
        f"state: [{color}]{record.state.value}[/{color}]",
        f"job:   {record.job_id}",
        f"out:   {record.out_dir}",
    ]
    for name, files in record.outputs.items():
        lines.append(f"{name}: {len(files)} file(s)")
    console.print(Panel("\n".join(lines), title=f"Run {record.run_id} · {record.tool}", expand=False))


def _engine(agent: bool):  # noqa: ANN202 - RAGChain | ChaperoneAgent, both have .invoke
    from chaperone.app import build_agent, build_rag_chain

    return build_agent() if agent else build_rag_chain()


def _render(answer: Answer) -> None:
    badge = "[green]grounded[/green]" if answer.grounded else "[yellow]ungrounded[/yellow]"
    console.print(
        Panel(answer.text, title=f"Chaperone · {badge} · conf {answer.confidence:.2f}", expand=False)
    )
    if answer.citations:
        table = Table(title="Sources", show_lines=False)
        table.add_column("#", justify="right", style="cyan")
        table.add_column("Source")
        table.add_column("Where", style="dim")
        for c in answer.citations:
            table.add_row(str(c.marker), c.title or c.source_uri, c.locator or "")
        console.print(table)


def _welcome() -> None:
    console.print(
        Panel(
            "[bold cyan]Chaperone-RAG[/bold cyan] — cited answers for protein design.\n"
            "Type a question, or 'exit' to quit.",
            expand=False,
        )
    )


if __name__ == "__main__":
    app()
