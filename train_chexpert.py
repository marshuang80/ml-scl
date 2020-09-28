from args import BaseTrainArgParser
from lightning import LightningChexpertModel
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping
from constants import *

seed_everything(6)

def main(args):

    # logger
    tb_logger = pl_loggers.tensorboard.TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.experiment_name,
    )
    wb_logger = pl_loggers.WandbLogger(
        name=args.trial_suffix,
        save_dir=args.log_dir,
        project=args.experiment_name
    )

    # early stop call back
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        strict=False,
        verbose=False,
        mode='min'
    )

    model = LightningChexpertModel(args)
    trainer = Trainer.from_argparse_args(
        args,
        logger=[tb_logger, wb_logger],
        early_stop_callback=early_stop
    )
    trainer.fit(model)


if __name__ == '__main__':
    parser = BaseTrainArgParser().get_parser()
    
    # add model specific args
    parser = LightningChexpertModel.add_model_specific_args(parser)

    # add all the available trainer options to argparse
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    # experiment name 
    args.experiment_name = f"{args.wandb_project_name}_{args.model_name}"
    args.trial_suffix = f"{args.optimizer}_lr{args.lr}_lrd{args.lr_decay}" + \
                        f"_wd{args.weight_decay}_rs{args.resize_shape}" + \
                        f"_cr{args.crop_shape}_ro{args.rotation_range}" + \
                        f"_gns{args.gaussian_noise_std}_gbr{args.gaussian_blur_radius}"

    main(args)