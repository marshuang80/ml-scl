import os 
import sys
sys.path.append(os.getcwd())

from .base_arg_parser import BaseTrainArgParser
from constants        import *



class ContrastiveTrainArgParser(BaseTrainArgParser):
    """Argument parser for CheXpert Training"""

    def __init__(self):
        super(ContrastiveTrainArgParser, self).__init__()

        # CheXpert specific arguments
        self.parser.add_argument("--save_freq", type=int, default=50)
        self.parser.add_argument('--match_type', type=str, default='all', choices=MATCH_TYPE)
        self.parser.add_argument("--temp", type=float, default=0.07)
        self.parser.add_argument("--log_dir", type=str, default="./contrastive_log")