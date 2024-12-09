import logging
class ConsoleLogger:
    def __init__(self, verbose):
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        self.setup_console_logger()


    def setup_console_logger(self):
        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(logging.INFO)
        self.console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(self.console_handler)
        if self.verbose:
            self.logger.setLevel(logging.DEBUG)
            self.logger.debug("Verbose mode enabled.")
    
    def info(self, message):
        self.logger.info(message)

    def debug(self, message):
        self.logger.debug(message)

    def error(self, message):
        self.logger.error(message)

    def warning(self, message):
        self.logger.warning(message)

    def critical(self, message):
        self.logger.critical(message)

    def exception(self, message):
        self.logger.exception(message)

    