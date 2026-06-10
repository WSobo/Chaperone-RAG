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


# --- helpers -----------------------------------------------------------------


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
