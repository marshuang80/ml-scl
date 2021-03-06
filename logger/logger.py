import wandb
import pandas as pd
import torch
import util

from constants                import *
from torch.utils.tensorboard  import SummaryWriter
from pathlib                  import Path

class Logger:

    def __init__(self, log_dir:str, metrics_name, args):
  
        # define relevant paths
        self.trial_tb_dir = Path(log_dir) / args.experiment_name / 'tensorboard'   
        self.log_file = Path(log_dir) / args.experiment_name / 'metrics'           
        self.ckpt_dir = Path(log_dir) / args.experiment_name / 'checkpoints'       
  
        # define summery writer
        self.writer = SummaryWriter(self.trial_tb_dir)
        hyperparameter_defaults = vars(args)
        if args.trial_suffix is not None:
            wandb.init(name=args.trial_suffix, config=hyperparameter_defaults, project=args.wandb_project_name)
        else:
            wandb.init(config=hyperparameter_defaults, project=args.wandb_project_name)

        # define experiment/trial file structure
        self._init_trial_dir()
        self.min_metrics = float('-inf') #gets updated later
        self.metrics_name = metrics_name
        self.conditions = CHEXPERT_COMPETITION_TASKS
  
    def log_dict(self, dict_values, step, split):
        """Log metrics onto tensorboard and wandb

        Args:
            dict_values (dict): dictionary of metrics
            step (int): interation step
            split (str): datasplit 
        """
        #write to wandb
        wandb.log(dict_values)
  
        # write to tensorboard
        for key, value in dict_values.items():
            self.writer.add_scalar(key, value, step)
  
    def log_image(self, img, step):
        """Log image into tensorboard """
        # get first image 
        img = img[0]
  
        # unnormalize first image
        img = util.unnormalize(img)

        img[img > 1] = 1
        img[img < 0] = 0 
 
        # log image
        self.writer.add_image('images', img, step)
  
    def log_iteration(self, dict_values, step, split):
        """
         Log all relavent metrics to log file. Should be a csv file that looks
         like: 
              epoch | itration | train_loss | ...
         """
        metrics_file = self.log_file / Path('metrics.csv')
  
        dict_values.update({'step':step})
        dict_values.update({'split':split})
        df = pd.DataFrame(dict_values)
  
        if Path(metrics_file).is_file() == False:      
          df.to_csv(metrics_file)      
        else:        
          df_old = pd.read_csv(metrics_file)
          df = df.append(df_old)
          df.to_csv(metrics_file)
  
    def save_checkpoint(self, model, metrics_dict, itr):
  
        metrics_list=[]
        for pathology in self.conditions:
          metrics_list.append(metrics_dict[pathology][self.metrics_name])
        current_mean = sum(metrics_list) / len(metrics_list)
  
        if self.min_metrics < current_mean:
          ckpt_dict = {'model_name': model.__class__.__name__, 
                       'model_args': model.module.cpu().args_dict(),
                       'model_state': model.module.cpu().state_dict()}
          
          ckpt_path = self.ckpt_dir / f"{model.__class__.__name__}_{itr}.pth"
          torch.save(ckpt_dict, ckpt_path)
          
    def _init_trial_dir(self):
        """structure the log directory for this trial"""
        if self.trial_tb_dir.is_dir() == False:
            self.trial_tb_dir.mkdir(parents=True, exist_ok=False) #if the if command works, exception is never raised
  
        if self.log_file.is_dir() == False:
            self.log_file.mkdir(parents=True, exist_ok=False) 
  
        if self.ckpt_dir.is_dir() == False:
            self.ckpt_dir.mkdir(parents=True, exist_ok=False)
