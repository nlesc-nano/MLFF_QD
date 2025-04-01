import argparse
import logging

from mlff_qd.utils.helpers import load_config, parse_args
from mlff_qd.utils.logging_utils import setup_logging
from mlff_qd.utils.training import main

if __name__ == '__main__':
    args = parse_args()
    setup_logging()
    logging.info(f"{'*' * 30} Started {'*' * 30}")
    print("Started")
    main(args)
