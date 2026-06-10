"""Back-compat entry point. Prefer the installed `chaperone` CLI.

    python main.py            # help
    python main.py chat       # interactive REPL
    python main.py ask "..."  # one-shot cited answer
"""

from chaperone.cli import app

if __name__ == "__main__":
    app()
