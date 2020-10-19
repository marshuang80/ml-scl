import transforms
from pytorch_lightning import loggers
from argparse import ArgumentParser
from data_module import CheXpertDataModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping
from pl_bolts.models.self_supervised import SimCLR
from pl_bolts.models.self_supervised.simclr import SimCLREvalDataTransform, SimCLRTrainDataTransform


seed_everything(6)

def main(args):

    # logger
    # TODO: 
    """
    tb_logger = pl_loggers.tensorboard.TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.experiment_name,
    )
    wb_logger = pl_loggers.WandbLogger(
        name=args.trial_suffix,
        save_dir=args.log_dir,
        project=args.experiment_name
    )
    """

    # early stop call back
    early_stop = EarlyStopping(
        monitor='train_loss',
        patience=5,
        strict=False,
        verbose=False,
        mode='min'
    )

    #train_dataloader = MyDataset(transforms=SimCLRTrainDataTransform())
    #val_dataloader = MyDataset(transforms=SimCLREvalDataTransform())

    # TODO: Implement CheXpert DataModule
    dm = CheXpertDataModule.from_argparse_args(args)
    dm.train_transforms = transforms.chexpert.SimCLRTrainDataTransform(args.crop_size)
    dm.val_transforms = transforms.chexpert.SimCLREvalDataTransform(args.crop_size)
    args.num_samples = len(dm.train_dataloader().dataset)

    # finetune in real-time
    # understand if this is nessasary
    """
    def to_device(batch, device):
        (x1, x2), y = batch
        x1 = x1.to(device)
        y = y.to(device)
        return x1, y
    online_eval = SSLOnlineEvaluator(z_dim=2048 * 2 * 2, num_classes=dm.num_classes)
    online_eval.to_device = to_device
    """

    model = SimCLR(**args.__dict__)
    #trainer = Trainer.from_argparse_args(args, logger=[tb_logger, wb_logger], callbacks=[online_eval]) # if using SSLOnline Evaluator
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, dm)

if __name__ == '__main__':

    # Datamodule arguments
    parser = ArgumentParser()
    parser.add_argument('--image_type', type=str, default='all')
    parser.add_argument('--uncertain', type=str, default='ignore')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=224)

    # trainer args
    parser = Trainer.add_argparse_args(parser)

    # model args
    parser = SimCLR.add_model_specific_args(parser)
    args = parser.parse_args()

    main(args)