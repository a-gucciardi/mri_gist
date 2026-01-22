import logging
from rich.logging import RichHandler

def setup_logger(verbose=False):
    """ Logging with rich output """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )
