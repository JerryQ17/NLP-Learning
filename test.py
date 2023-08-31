import logging
from sys import stdout

from src import *

if __name__ == "__main__":
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_stream_handler = logging.StreamHandler(stdout)
    # root_file_handler = logging.FileHandler(r'.\log\log.txt', mode='w')
    root_formatter = logging.Formatter('[%(levelname)s] %(asctime)s %(name)s: %(message)s')
    root_stream_handler.setFormatter(root_formatter)
    # root_file_handler.setFormatter(root_formatter)
    root_logger.addHandler(root_stream_handler)
    # root_logger.addHandler(root_file_handler)

    t = word2vec_lstm()
