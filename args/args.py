import os
import sys
import torch
import argparse
sys.path.append(os.getcwd())
import constants


class TrainArgParser:
    '''Base training argument parser
    Shared with CheXpert and Contrastive loss Training
    '''

    def __init__(self):
        self.parser = argparse.ArgumentParser(description = "Pneumothorax Chest Xrays")

        #dataset
    def parse_args(self):
        args = self.parser.parse_args()
        return args