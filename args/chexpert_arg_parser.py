import os 
import sys
sys.path.append(os.getcwd())

from .base_arg_parser import BaseTrainArgParser
from constants        import *



class CheXpertTrainArgParser(BaseTrainArgParser):
    """Argument parser for CheXpert Training"""

    def __init__(self):
        super(CheXpertTrainArgParser, self).__init__()

        # CheXpert specific arguments
        self.parser.add_argument("--threshold", type=float, default=0.5)
        self.parser.add_argument("--log_dir", type=str, default="./chexpert_log")
        self.parser.add_argument("--eval_metrics", type=str, default="auroc", choices=CHEXPERT_EVAL_METRICS)
        self.parser.add_argument("--ckpt_path", type=str, default=None)