from __future__ import annotations
import logging
import logging.handlers as handlers
from typing import Any


import Code.Constants as c

class Logger(object):
    @staticmethod
    def __initialize(logger_root_path: str):
        """
        Initializes the logger
        :param logger_root_path: Logger root path
        :return: Logger
        """
        __logger = logging.getLogger(__name__)
        
        # Create the handlers
        c_handler = logging.StreamHandler()
        f_handler = handlers.RotatingFileHandler(logger_root_path, maxBytes=2000000, backupCount=20)
        c_handler.setLevel(logging.DEBUG)
        f_handler.setLevel(logging.DEBUG)
        
        # Create formatters and add it to the handlers
        c_format = logging.Formatter(c.LOGGING_FORMAT)
        f_format = logging.Formatter(c.LOGGING_FORMAT)
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        # Add handlers to the logger
        if (__logger.hasHandlers()):
            __logger.handlers.clear()
        __logger.addHandler(c_handler)
        __logger.addHandler(f_handler)
        __logger.setLevel(logging.DEBUG)
        return __logger

    @staticmethod
    def getLogger(logger_root_path: str = c.MOVIELENS_100K_LOGGING_PATH):
        """
        Gets the logger instance
        :param logger_root_path: Logger root path
        :return: Logger
        """
        logger = Logger.__initialize(logger_root_path)
        return logger