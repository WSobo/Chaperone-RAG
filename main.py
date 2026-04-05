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

    # Load prompt template
    with open('configs/agent_prompts.yaml', 'r') as f:
        prompts = yaml.safe_load(f)

    # Force a base document load if none exist (fallback from legacy memory API)
    system_setup = prompts['system_setup'].format(documentation=memory.get_docs())
    engine.chat(system_setup)
    
    logger.info("System setup complete. Chaperone Agent is ready for commands.")
    console.print("\n[bold green]Chaperone is ready. Type your biological questions below.[/bold green]")

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

            logger.info("Thinking... searching RAG context...")
            relevant_ctx = memory.search_context(user_question)
            
            # Simple prompt injection approach for retrieved context
            augmented_prompt = f"User Question: {user_question}\n\nRelevant Context from DB:\n{relevant_ctx}"
            
            response = engine.chat(augmented_prompt)
            console.print("\n[bold blue]--- CHAPERONE ---[/bold blue]")
            console.print(response)

        except (KeyboardInterrupt, EOFError):
             logger.info("Interrupt received. Shutting down...")
             break

if __name__ == "__main__":
    main()
