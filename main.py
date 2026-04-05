import yaml
import sys
import argparse
from rich.console import Console
from rich.panel import Panel
from chaperone.engine import GemmaEngine
from chaperone.memory import RAGMemory
from chaperone.utils.logger import logger

console = Console()

def print_welcome():
    welcome_text = """[bold cyan]╔══════════════════════════════════════════════════╗
║                                                  ║
║  🧬 Chaperone-RAG: Computational Biology Agent 🤖 ║
║               powered by Gemma 4                 ║
║                                                  ║
╚══════════════════════════════════════════════════╝[/bold cyan]"""
    console.print(Panel(welcome_text, title="Starting", expand=False))

def main():
    parser = argparse.ArgumentParser(description="Chaperone-RAG CLI Interface")
    parser.add_argument("--test", action="store_true", help="Run a quick test query and exit.")
    parser.add_argument("--ingest", nargs="+", help="List of URLs to force-ingest into the Chroma DB.")
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run startup warmup prompt (slower startup).",
    )
    args = parser.parse_args()

    print_welcome()
    logger.info("Booting up Chaperone engine & memory subsystems...")

    engine = GemmaEngine()
    memory = RAGMemory()

    # If --ingest was passed, ingest the web paths into the DB
    if args.ingest:
        logger.info(f"Command line ingest triggered for {len(args.ingest)} URLs...")
        memory.ingest_urls(args.ingest)

    # Automatically ingest any PDFs dropped into the data/papers folder
    memory.ingest_pdfs(folder_path="data/papers")

    if args.warmup:
        logger.info("Warmup enabled. Running startup system prompt...")
        with open("configs/agent_prompts.yaml", "r") as f:
            prompts = yaml.safe_load(f)
        # Optional warmup: expensive on large models, so keep opt-in.
        system_setup = prompts["system_setup"].format(documentation=memory.get_docs())
        engine.chat(system_setup)
    else:
        logger.info("Skipping startup warmup for fast launch. Use --warmup to enable.")
    
    logger.info("System setup complete. Chaperone Agent is ready for commands.")
    console.print("\n[bold green]Chaperone is ready. Type your biological questions below.[/bold green]")
    console.print("[dim]Tip: /personas to list personas, /persona <name> to pin one, /persona auto to reset.[/dim]")

    persona_override = None

    if args.test:
        test_query = "What is RFdiffusion?"
        console.print(f"\n[bold yellow]Test Query:[/bold yellow] {test_query}")
        # Note: True RAG intercepts the user query, searches, and injects context.
        # This is a very simple pass-through to Gemma for now.
        response = engine.chat(test_query)
        console.print(f"\n[bold blue]--- CHAPERONE ---[/bold blue]\n{response}")
        sys.exit(0)

    # Interactive Loop
    while True:
        try:
            user_question = console.input("\n[green]Ask Chaperone >[/green] ")
            
            if user_question.lower() in ['exit', 'quit', 'q']:
                logger.info("Gracefully shutting down Chaperone...")
                break

            if not user_question.strip():
                continue

            if user_question.startswith("/"):
                command = user_question.strip()
                if command in {"/personas", "/persona list"}:
                    names = ", ".join(engine.router.available_personas())
                    console.print(f"[cyan]Available personas:[/cyan] {names}")
                    continue
                if command.startswith("/persona "):
                    value = command.split(" ", 1)[1].strip()
                    if value.lower() == "auto":
                        persona_override = None
                        console.print("[cyan]Persona routing set to automatic.[/cyan]")
                        continue

                    resolved = engine.router.resolve_persona_name(value)
                    if not resolved:
                        console.print("[red]Unknown persona.[/red] Use /personas to see valid names.")
                        continue

                    persona_override = resolved
                    console.print(f"[cyan]Pinned persona:[/cyan] {persona_override}")
                    continue

                if command == "/help":
                    console.print("[cyan]Commands:[/cyan] /personas, /persona <name>, /persona auto, exit")
                    continue

            persona = persona_override if persona_override else engine.router.route_intent(user_question)
            if persona_override:
                logger.info(f"Using pinned persona override: [bold cyan]{persona}[/bold cyan]")
            if persona == "friendly_chatter":
                logger.info("Friendly chat mode: skipping RAG retrieval.")
                augmented_prompt = user_question
            else:
                logger.info("Thinking... searching RAG context...")
                relevant_ctx = memory.search_context(user_question)

                # Simple prompt injection approach for retrieved context
                augmented_prompt = f"User Question: {user_question}\n\nRelevant Context from DB:\n{relevant_ctx}"
            
            response = engine.chat(augmented_prompt, forced_persona=persona)
            console.print("\n[bold blue]--- CHAPERONE ---[/bold blue]")
            console.print(response)

        except (KeyboardInterrupt, EOFError):
             logger.info("Interrupt received. Shutting down...")
             break

if __name__ == "__main__":
    main()
