import sys
import logging
from pathlib import Path


class Logger:
    def __init__(
        self,
        log_file="results.log",
        level=logging.INFO,
        console=True,
        overwrite=True,
    ):
        self.log_file = Path(log_file)
        self.level = level
        self.console = console

        self.logger = logging.getLogger(str(self.log_file.resolve()))
        self.logger.setLevel(level)
        self.logger.propagate = False

        for h in self.logger.handlers[:]:
            self.logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        if console:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(level)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

        file_mode = "w" if overwrite else "a"
        fh = logging.FileHandler(self.log_file, mode=file_mode, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def debug(self, msg: str):
        self.logger.debug(msg)

    def get_logger(self):
        return self.logger
