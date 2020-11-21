import transforms
from pytorch_lightning import loggers
from argparse import ArgumentParser
from data_module import CheXpertDataModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping
from pl_bolts.models.self_supervised import SwAV
from pl_bolts.models.self_supervised.swav import SwAVTrainDataTransform, SwAVEvalDataTransform


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
        monitor='train_loss',
        patience=5,
        strict=False,
        verbose=False,
        mode='min'
    )

    dm = CheXpertDataModule.from_argparse_args(args)
    # TODO: use custom data transfrom from transforms.chexpert.SwAVvalDataTransform 
    # The custom transforms should be tailored for medical imaging
    dm.train_transforms = SwAVTrainDataTransform(
        size_crops = [224, 96],
        nmb_crops = [2, 4],
        min_scale_crops = [0.33, 0.10],
        max_scale_crops = [1, 0.33],
        gaussian_blur = True,
    )
    dm.val_transforms = SwAVEvalDataTransform(
        size_crops = [224, 96],
        nmb_crops = [2, 4],
        min_scale_crops = [0.33, 0.10],
        max_scale_crops = [1, 0.33],
        gaussian_blur = True,
    )
    args.num_samples = len(dm.train_dataloader().dataset)

    # TODO: maybe use online eval?

    # Define model
    args.maxpool1 = True
    args.first_conv = True
    args.lars_wrapper = True
    args.online_ft = False
    model = SwAV(**args.__dict__)
    trainer = Trainer.from_argparse_args(args, logger=[tb_logger, wb_logger])
    trainer.fit(model, dm)

if __name__ == '__main__':

    # Datamodule arguments
    parser = ArgumentParser()
    parser.add_argument('--image_type', type=str, default='all')
    parser.add_argument('--uncertain', type=str, default='ignore')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=224)

    # Trainer arguments
    parser.add_argument('--data_dir', type=str, default='/data/selfsupervision/CheXpert/CheXpert-v1.0/')
    parser.add_argument('--precision', type=int, default=16)
    parser.add_argument('--weights_save_path', type=str, default='/data/ckpt')
    parser.add_argument('--auto_lr_find', type=str, default='lr')
    parser.add_argument('--tpu_cores', type=int, default=8)

    # model args
    parser = SwAV.add_model_specific_args(parser)
    args = parser.parse_args()

    main(args)
