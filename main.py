import argparse
import json
import random
from pathlib import Path
from datetime import datetime
import os
from model import model_dict
from datasets import dataset_dict
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, dataloader
from torch.utils.data.distributed import DistributedSampler
from engine import train_one_epoch, evaluate
from torch.utils.tensorboard import SummaryWriter
import utils.misc as utils
#from adan import Adan
def get_args_parse():
    parser = argparse.ArgumentParser('Dense NeRV', add_help=False)

    parser.add_argument('--cfg_path', default='', type=str, help='path to specific cfg yaml file path')
    parser.add_argument('--output_dir', default='', type=str, help='path to save the log and other files')
    parser.add_argument('--time_str', default='', type=str, help='just for tensorboard dir name')
    #parser.add_argument('--device', default='cpu', help='device to use for training / testing')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--port', default=29500, type=int, help='port number')
    parser.add_argument('--rank', default=0, type=int, help='port number')
    parser.add_argument('--max_grad_norm', type=float, default=0.0, help='if the l2 norm is large than this hyper-parameter, then we clip the gradient  (default: 0.0, no gradient clip)')
    parser.add_argument('--weight_decay', type=float, default=0.02,  help='weight decay, similar one used in AdamW (default: 0.02)')
    parser.add_argument('--opt_eps', default=None, type=float, metavar='EPSILON', help='optimizer epsilon to avoid the bad case where second-order moment is zero (default: None, use opt default 1e-8 in adan)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA', help='optimizer betas in Adan (default: None, use opt default [0.98, 0.92, 0.99] in Adan)')
    parser.add_argument('--no_prox', action='store_true', default=False, help='whether perform weight decay like AdamW (default=False)')

    return parser


def main(args):
	utils.init_distributed_mode(args)
	print('git:\n {}\n'.format(utils.get_sha()))

	# get cfg yaml file
	cfg = utils.load_yaml_as_dict(args.cfg_path)
	# dump the cfg yaml file in output dir
	utils.dump_cfg_yaml(cfg, args.output_dir)
	print(cfg)

	device = torch.device(args.device)
	seed = cfg['seed']

	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)

	model = model_dict[cfg['model']['model_name']](cfg=cfg['model'])
	model.to(device)
	model_without_ddp = model

	if args.rank in [0, None]:
		for n_, p in model.named_parameters():
			print('Num: ', p.data.nelement(), ' name: ', n_)
		params = sum([p.data.nelement() for p in model.parameters()]) / 1e6
		params_de = sum([p.data.nelement() for n__, p in model.named_parameters() if 'dec' in n__]) / 1e6
		params_de_ = sum([p.data.nelement() for n__, p in model.named_parameters() if 'enc' not in n__]) / 1e6
		params_en = sum([p.data.nelement() for n__, p in model.named_parameters() if 'enc' in n__]) / 1e6
		print(f'{args}\n {model}\n Model Params: {params}M')
		print(f'Encoder Params: {params_en}M')
		print(f'Decoder Params: {params_de}M')
		print(f'Decoder minus Params: {params_de_}M')
	
	writer = SummaryWriter(os.path.join(args.output_dir, 'tensorboard_{}'.format(args.time_str)))
	img_transform = transforms.ToTensor()

	dataset_train = dataset_dict[cfg['dataset_type']](main_dir=cfg['dataset_path'], transform=img_transform,height=cfg['model']['height'],width=cfg['model']['width'], train=True)
	dataset_val = dataset_dict[cfg['dataset_type']](main_dir=cfg['dataset_path'], transform=img_transform, height=cfg['model']['height'], width=cfg['model']['width'], train=False)

	sampler_train = DistributedSampler(dataset_train) if args.distributed else None
	sampler_val = DistributedSampler(dataset_val) if args.distributed else None

	dataloader_train = DataLoader(
		dataset_train, batch_size=cfg['train_batchsize'], shuffle=(sampler_train is None), num_workers=cfg['workers'], 
		pin_memory=True, sampler=sampler_train, drop_last=True, worker_init_fn=utils.worker_init_fn
	)
	dataloader_val = DataLoader(
		dataset_val, batch_size=cfg['val_batchsize'], shuffle=False, num_workers=cfg['workers'], 
		pin_memory=True, sampler=sampler_val, drop_last=False, worker_init_fn=utils.worker_init_fn
	)

	datasize = len(dataset_train)
	param_dicts = [
		{
			"params": [p for n, p in model_without_ddp.named_parameters() if p.requires_grad],
			"lr": cfg['optim']['lr'],
		}
	]

	optim_cfg = cfg['optim']
	optimizer = optim.Adam(param_dicts, lr=optim_cfg['lr'], betas=(optim_cfg['beta1'], optim_cfg['beta2']))

	if args.distributed:
		model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
		model_without_ddp = model.module

	output_dir = Path(args.output_dir)
	train_best_psnr, train_best_msssim, val_best_psnr, val_best_msssim = [torch.tensor(0) for _ in range(4)]

	print('Start training')
	print(' Seed : ',  cfg['seed'])
	print(' Learning rate: ', cfg['optim']['lr'])
	print(' Dataset : ',  cfg['dataset_path'])
	log_300 = []
	print('optimizer: ' , optimizer)


	start_time = datetime.now()
	for epoch in range(cfg['epoch']):
		if args.distributed:
			sampler_train.set_epoch(epoch)
		train_stats = train_one_epoch(
			model, dataloader_train, optimizer, device, epoch, cfg, args, datasize, start_time, writer
		)

		train_best_psnr = train_stats['train_psnr'][-1] if train_stats['train_psnr'][-1] > train_best_psnr else train_best_psnr
		train_best_msssim = train_stats['train_msssim'][-1] if train_stats['train_msssim'][-1] > train_best_msssim else train_best_msssim
		if args.rank in [0, None]:
			print_str = '\ttraining: current: {:.2f}\t best: {:.2f}\t msssim_best: {:.4f}\t'.format(train_stats['train_psnr'][-1].item(), 
			train_best_psnr.item(), train_best_msssim.item())
			print(print_str, flush=True)
		
		checkpoint_paths = [output_dir / 'checkpoint.pth']  # save one per epoch
		for checkpoint_path in checkpoint_paths:
			utils.save_on_master({
				'model': model_without_ddp.state_dict(),
				'optimizer': optimizer.state_dict(),
				'epoch': epoch,
				'config': cfg,
				'train_best_psnr': train_best_psnr,
				'train_best_msssim': train_best_msssim,
				'val_best_psnr': val_best_psnr,
				'val_best_msssim': val_best_msssim,
			}, checkpoint_path)
		if epoch == cfg['epoch']-1 :
			print("Encoding time: " + str(datetime.now() - start_time))		

		if (epoch + 1) % cfg['eval_freq'] == 0 or epoch > cfg['epoch'] - 2:
			val_stats = evaluate(model, dataloader_val, device, cfg, args, save_images=False, Epoch=epoch)
			val_best_psnr = val_stats['val_psnr'][-1] if val_stats['val_psnr'][-1] > val_best_psnr else val_best_psnr
			val_best_msssim = val_stats['val_msssim'][-1] if val_stats['val_msssim'][-1] > val_best_msssim else val_best_msssim

			if (epoch + 1) % 50 ==0:
				val_300 = [val_stats['val_psnr'][-1].item(), val_stats['val_msssim'][-1].item()]
				log_300.append(val_300)
				for i in range(len(log_300)):
					print('======PSNR/SSIM in E', str(i*50), ': ',log_300[i][0], log_300[i][1], '======')

			val_best_psnr = val_stats['val_psnr'][-1] if val_stats['val_psnr'][-1] > val_best_psnr else val_best_psnr
			val_best_msssim = val_stats['val_msssim'][-1] if val_stats['val_msssim'][-1] > val_best_msssim else val_best_msssim
			if args.rank in [0, None]:
				print_str = f'Eval best_PSNR at epoch{epoch+1}:'
				print_str += '\tevaluation: current: {:.2f}\tbest: {:.2f} \tbest_msssim: {:.4f}'.format(
					val_stats['val_psnr'][-1].item(), val_best_psnr, val_best_msssim)
				print(print_str)

	print("Training complete in: " + str(datetime.now() - start_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('E-NeRV training and evaluation script', parents=[get_args_parse()])
    args = parser.parse_args()

    assert args.cfg_path is not None, 'Need a specific cfg path!'
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

