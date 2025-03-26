import math
import random
import numpy as np
from tqdm import tqdm
import omegaconf
from omegaconf import OmegaConf
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from dataset.dataset import PreTrainingDataset, OfflineDataset, OnlineTrainingDataset
import os
import os.path as opt
import sys
sys.path.append('./model')
from model.vit_mae import ViT
from model.stmae import STMAE, Encoder
from utils.stmae_utils import stmae_training_loop
from model.stgcn import STGCN
from utils.stgcn_utils import valid_one_epoch, stgcn_offline_training_loop, stgcn_online_training_loop
from anatomical_loss import AnatomicalLoss, get_data_stats, plot_data_stats
import wandb


def seed_everything(seed):
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	random.seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Train STMAE')
	parser.add_argument('--cfg_path', default='configs/train_STMAE.yaml', help='Path to the train.yaml config')
	args = parser.parse_args()
	## configs
	cfg_path = args.cfg_path
	args = OmegaConf.load(cfg_path)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	seed_everything(args.seed)

	print('\n\n', '='*15, 'ARGUMENTS.', '='*15)
	for arg, val in args.__dict__['_content'].items():
		if isinstance(val, omegaconf.nodes.AnyNode):
			print('> {}: {}'.format(arg, val))
		else:
			print('> {}'.format(arg))
			for arg_, val_ in val.items():
				print('\t- {}: {}'.format(arg_, val_))


	## FIRST PHASE: PRE-TRAINING
	print('\n\n', '='*15, 'FIRST PHASE: PRE-TRAINING', '='*15)

	## DATA SETS & LOADERS
	print('\nLOADING DATA....')
	data_args = args.data
	stmae_args = args.stmae
	info = {
		'dataset': args.dataset,
		'n_joints': data_args.n_joints,
		'mean': data_args.mean,
		'std': data_args.std,
		'joints_connections': data_args.joints_connections,
		'label_map': data_args.label_map,
	}


	# Initialize WandB
	wandb.init(
		project="STMAE-Parkinson",
		name=args.exp_name,  # Use the experiment name from the config
		config={
			# Experiment Config
			"experiment_name": args.exp_name,
			"seed": args.seed,
			"dataset": args.dataset,
			"save_folder_path": args.save_folder_path,
			
			# Data Config
			"train_data_dir": args.data.train_data_dir,
			"val_data_dir": args.data.val_data_dir,
			"test_data_dir": args.data.test_data_dir,
			"step": args.data.step,
			"normalize": args.data.normalize,
			"mean": args.data.mean,
			"std": args.data.std,
			"n_joints": args.data.n_joints,
			"label_map": args.data.label_map,
			"joints_connections": args.data.joints_connections,
			
			# STMAE Config
			"stmae_num_joints": args.stmae.num_joints,
			"stmae_coords_dim": args.stmae.coords_dim,
			"stmae_encoder_embed_dim": args.stmae.encoder_embed_dim,
			"stmae_encoder_depth": args.stmae.encoder_depth,
			"stmae_num_heads": args.stmae.num_heads,
			"stmae_mlp_dim": args.stmae.mlp_dim,
			"stmae_decoder_dim": args.stmae.decoder_dim,
			"stmae_decoder_depth": args.stmae.decoder_depth,
			"stmae_window_size": args.stmae.window_size,
			"stmae_masking_strategy": args.stmae.masking_strategy,
			"stmae_spatial_masking_ratio": args.stmae.spatial_masking_ratio,
			"stmae_temporal_masking_ratio": args.stmae.temporal_masking_ratio,
			"stmae_anatomical_loss": args.stmae.anatomical_loss,
			"stmae_root_index": args.stmae.root_index,
			"stmae_num_epochs": args.stmae.num_epochs,
			"stmae_lr": args.stmae.lr,
			"stmae_weight_decay": args.stmae.weight_decay,
			"stmae_batch_size": args.stmae.batch_size,

		},
	)


	train_set = PreTrainingDataset(data_dir=data_args.train_data_dir,
									 window_size=stmae_args.window_size,
									 step=data_args.step,
									 normalize=data_args.normalize,
									 info=info,
									 random_rot=False)

	valid_set = PreTrainingDataset(data_dir=data_args.test_data_dir,
									 window_size=stmae_args.window_size,
									 step=data_args.step,
									 info=info,
									 normalize=data_args.normalize)

	print('# Train: {}, # Valid: {}'.format(len(train_set), len(valid_set)))
	train_loader = DataLoader(train_set, batch_size=stmae_args.batch_size, shuffle=True)
	valid_loader = DataLoader(valid_set, batch_size=stmae_args.batch_size, shuffle=False)

	print('\nBUILDING MODEL....')
	## MODEL & OPTIMIZER
	if stmae_args.anatomical_loss:

		stats_set = PreTrainingDataset(data_dir=data_args.train_data_dir,
									 window_size=stmae_args.window_size,
									 step=stmae_args.window_size, ## overlook redondante frames
									 normalize=data_args.normalize,
									 info=info)
		stats_loader = DataLoader(stats_set, batch_size=stmae_args.batch_size, shuffle=True)

		angles_stats, lengths_stats = get_data_stats(stats_loader, stmae_args.root_index)
		os.makedirs(opt.join(args.save_folder_path, args.dataset, args.exp_name), exist_ok=True)
		plot_data_stats(angles_stats, lengths_stats, 
						fname=opt.join(args.save_folder_path, args.dataset, args.exp_name, 'stats.png'))
		anatomical_loss = AnatomicalLoss(angles_stats['min'].to(device),
										 angles_stats['max'].to(device),
										 lengths_stats['min'].to(device), 
										 lengths_stats['max'].to(device), 
										 stmae_args.root_index)
	else:
		anatomical_loss = None

	encoder = Encoder(
				patch_num=stmae_args.num_joints*stmae_args.window_size,
				patch_dim=stmae_args.coords_dim,
				window_size=stmae_args.window_size,
				num_classes=stmae_args.coords_dim,
				dim=stmae_args.encoder_embed_dim,
				depth=stmae_args.encoder_depth,
				heads=stmae_args.num_heads,
				mlp_dim=stmae_args.mlp_dim ,
				pool = 'cls',
				# channels = 3,
				dim_head = 64,
				dropout = 0.,
				emb_dropout = 0.
			)

	stmae = STMAE(
				encoder=encoder,
				decoder_dim=stmae_args.decoder_dim,
				decoder_depth=stmae_args.decoder_depth,
				masking_strategy=stmae_args.masking_strategy,
				spatial_masking_ratio=stmae_args.spatial_masking_ratio,
				temporal_masking_ratio=stmae_args.temporal_masking_ratio,
				anatomical_loss=anatomical_loss
			).to(device)

	optimizer = optim.AdamW(stmae.parameters(), lr=stmae_args.lr)
	scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

	n_params = sum(p.numel() for p in stmae.parameters() if p.requires_grad)
	print("Number of trainable parameters of STMAE: ", n_params)

	stmae_training_loop(stmae, train_loader, valid_loader, device, optimizer, scheduler, wandb, args)
 	
	# Log train-validation loss plot to WandB
	sim_folder = f'{args.save_folder_path}/{args.dataset}/{args.exp_name}'
	wandb.log({"train_loss_plot": wandb.Image(f'{sim_folder}/train_validation_loss_plot.png')})

	# Save the trained model artifact
	wandb.save(f'{sim_folder}/model.pth')
	wandb.finish()


	