import logging
from rich.logging import RichHandler
from rich.console import Console

console = Console()

def setup_logger(name: str = "chaperone", level: int = logging.INFO) -> logging.Logger:
    """Sets up a fancy rich logger for the Chaperone RAG project."""
    logger = logging.getLogger(name)
    
    # Avoid adding multiple handlers if setup is called multiple times
    if not logger.handlers:
        logger.setLevel(level)
        handler = RichHandler(
            console=console, 
            rich_tracebacks=True, 
            markup=True,
            show_time=True,
            show_path=False
        )
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger

logger = setup_logger()
