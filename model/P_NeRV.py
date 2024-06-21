# ---- Pyramidal  NeRV ---- #
# ---- QiZhao CVPR2024 ---- #
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_utils import ActivationLayer
from torch import nn
from timm.models.layers import DropPath

'===============================ConvNeXT======================================='
class ConvNeXtBlock(nn.Module):
	def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
		super().__init__()
		self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
		self.norm = LayerNorm(dim, eps=1e-6)
		self.pwconv1 = nn.Linear(dim, 4 * dim)
		self.act = nn.GELU()
		self.pwconv2 = nn.Linear(4 * dim, dim)
		self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
									requires_grad=True) if layer_scale_init_value > 0 else None
		self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

	def forward(self, x):
		input = x
		x = self.dwconv(x)
		x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
		x = self.norm(x)
		x = self.pwconv1(x)
		x = self.act(x)
		x = self.pwconv2(x)
		if self.gamma is not None:
			x = self.gamma * x
		x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

		x = input + self.drop_path(x)
		return x

class LayerNorm(nn.Module):
	def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
		super().__init__()
		self.weight = nn.Parameter(torch.ones(normalized_shape))
		self.bias = nn.Parameter(torch.zeros(normalized_shape))
		self.eps = eps
		self.data_format = data_format
		if self.data_format not in ["channels_last", "channels_first"]:
			raise NotImplementedError 
		self.normalized_shape = (normalized_shape, )

	def forward(self, x):
		if self.data_format == "channels_last":
			return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
		elif self.data_format == "channels_first":
			u = x.mean(1, keepdim=True)
			s = (x - u).pow(2).mean(1, keepdim=True)
			x = (x - u) / torch.sqrt(s + self.eps)
			x = self.weight[:, None, None] * x + self.bias[:, None, None]
			return x
'=======================================ConvNeXT======================================='
class KFc_bias(nn.Module):
	def __init__(self, in_batch=1, in_height=2, in_width=4, out_height=4, out_width=8, channels=4):
		super().__init__()
		self.in_b = in_batch
		self.in_h = in_height
		self.in_w = in_width
		self.c = channels
		self.out_h = out_height
		self.out_w = out_width

		self.w_L_ = torch.normal(0, 1/self.in_h,  (self.c, self.out_h, self.in_h))
		self.w_R_ = torch.normal(0, 1/self.out_w, (self.c, self.in_w, self.out_w))
		nn.init.kaiming_normal_(self.w_L_, mode='fan_out', nonlinearity='relu')
		nn.init.kaiming_normal_(self.w_R_, mode='fan_out', nonlinearity='relu')

		self.w_L = nn.Parameter(self.w_L_.repeat(self.in_b,1,1,1))
		self.w_R = nn.Parameter(self.w_R_.repeat(self.in_b,1,1,1))

		self.b_h = nn.Parameter(torch.zeros(self.out_h, 1))
		self.b_w = nn.Parameter(torch.zeros(1, self.out_w))
		self.b_c = nn.Parameter(torch.zeros(self.c, 1))

	def forward(self, x):
		b_ = self.b_h @ self.b_w
		b__ = b_.reshape(1, self.out_h*self.out_w)
		_ = self.b_c @ b__
		__ = _.reshape(self.c, self.out_h, self.out_w)
		b = __.repeat(self.in_b, 1, 1, 1)

		_ = torch.matmul(self.w_L, x)
		return torch.matmul(_, self.w_R) + b


def NeRV_MLP(dim_list, act='relu', bias=True):
	act_fn = ActivationLayer(act)
	fc_list = []
	for i in range(len(dim_list) - 1):
		fc_list += [nn.Linear(dim_list[i], dim_list[i+1], bias=bias), act_fn]
	return nn.Sequential(*fc_list)


class NeRV_CustomConv(nn.Module):
	def __init__(self, **kargs):
		super(NeRV_CustomConv, self).__init__()

		ngf, new_ngf, kernel, stride, padding = kargs['ngf'], kargs['new_ngf'], kargs['kernel'], kargs['stride'], kargs['padding']
		self.conv = nn.Conv2d(ngf, new_ngf * stride * stride, kernel, 1,  padding)
		self.up_scale = nn.PixelShuffle(stride)

	def forward(self, x):
		out = self.conv(x)
		return self.up_scale(out)

class HNeRVBlock(nn.Module):
	def __init__(self, **kargs):
		super().__init__()
		self.conv = NeRV_CustomConv(ngf=kargs['ngf'], new_ngf=kargs['new_ngf'], kernel=kargs['kernel'], stride=kargs['stride'],  
				padding=kargs['padding'])
		self.act = ActivationLayer(kargs['act'])

	def forward(self, x):
		return self.act(self.conv(x))

class P_NeRV_Generator(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg

		self.c1_dim = cfg['c1'] #64
		self.d_dim  = cfg['d']  #16

		# BUILD CONV LAYERS
		self.dec_layers, self.enc_layers, self.dec_head_layers = [nn.ModuleList() for _ in range(3)]
		self.dec_exc_layers = nn.ModuleList()

		self.dec_shortcuts = nn.ModuleList()
		self.dec_bsm_z = nn.ModuleList()
		self.dec_bsm_r = nn.ModuleList()
		self.dec_bsm_h = nn.ModuleList()

		self.enc_diff_layers = nn.ModuleList()
		self.enc_diff_ebd_layer = nn.ModuleList()
		self.dec_diff_exc_layer = nn.ModuleList()

		ngf = self.d_dim

		#----------------ENCODER----------------#
		for k, stride in enumerate(cfg['encoder_list']):
			if k == 0:
				c0 = 3
			else:
				c0 = self.c1_dim
			self.enc_layers.append(nn.Conv2d(c0, self.c1_dim, kernel_size=cfg['encoder_list'][k], stride=cfg['encoder_list'][k]))
			self.enc_layers.append(LayerNorm(self.c1_dim, eps=1e-6, data_format="channels_first"))
			self.enc_layers.append(ConvNeXtBlock(dim=self.c1_dim))

			if k<len(cfg['diff_enc_list']):
				if k == 0:
					c0 = 6
				else:
					c0 = self.c1_dim
				self.enc_diff_layers.append(nn.Conv2d(c0, self.c1_dim, kernel_size=cfg['diff_enc_list'][k],
						stride=cfg['diff_enc_list'][k]))
				self.enc_diff_layers.append(LayerNorm(self.c1_dim, eps=1e-6, data_format="channels_first"))
				self.enc_diff_layers.append(ConvNeXtBlock(dim=self.c1_dim))
				self.enc_diff_ebd_layer = nn.Conv2d(self.c1_dim, 2, kernel_size=1, stride=1)
			self.enc_ebd_layer = nn.Conv2d(self.c1_dim, self.d_dim, kernel_size=1, stride=1)

		#----------------DECODER----------------#
		self.dec_exc_layers.append(HNeRVBlock(ngf=self.d_dim, new_ngf=self.d_dim, kernel=1, stride=10, padding=0, act=cfg['act']))

		ngf = int(self.d_dim)
		ngf__ = 2

		new_ngf = int(cfg['kfc_h_w_c'][2])
		ngf_ = new_ngf + ngf
		new_h , new_w = cfg['kfc_h_w_c'][0], cfg['kfc_h_w_c'][1]

		if len(cfg['diff_enc_list'])==4:
			in_h, in_w= 10, 20
		elif len(cfg['diff_enc_list'])==3:
			in_h, in_w= 40, 80
		else:
			in_h, in_w= 2, 4

		_s = 1
		for i, stride in enumerate(cfg['kfc_stride']):
			new_h , new_w = new_h*stride, new_w*stride
			if i != len(cfg['kfc_stride'])-1:
				if i ==0 and len(cfg['diff_enc_list'])==3:
					self.dec_shortcuts.append(nn.Identity())
					self.dec_shortcuts.append(nn.Identity())
					self.dec_shortcuts.append(nn.Identity())
				else:
					self.dec_shortcuts.append(KFc_bias(in_height=in_h, in_width=in_w, out_height=new_h, out_width=new_w, channels=ngf__))
					self.dec_shortcuts.append(nn.BatchNorm2d(ngf__, track_running_stats=True))
					self.dec_shortcuts.append(ActivationLayer(cfg['act']))

				self.dec_bsm_z.append(nn.Conv2d(new_ngf, new_ngf, kernel_size=3, stride=1, padding=1))
				self.dec_bsm_r.append(nn.Conv2d(ngf__, new_ngf, kernel_size=3, stride=1, padding=1))
				self.dec_bsm_h.append(nn.Conv2d(new_ngf, new_ngf, kernel_size=3, stride=1, padding=1))

			self.dec_layers.append(HNeRVBlock(ngf=ngf, new_ngf=new_ngf, kernel=3, stride=stride, padding=1, act=cfg['act']))
			ngf = new_ngf
		self.dec_head_layers = nn.Conv2d(new_ngf, 3, 3, 1, 1)

	def forward(self, data):
		content_embedding = data['img_gt']
		content_p  = data['img_p']
		content_f  = data['img_f']
		content_gt = data['img_gt']

		#----------------ENCODER----------------#
		diff_p = content_gt - content_p
		diff_f = content_f - content_gt
		diff =  torch.stack([diff_p, diff_f], dim=2)
		diff = diff.view(diff.size(0),-1,diff.size(-2),diff.size(-1))

		for diff_enc_layer in self.enc_diff_layers:
			diff = diff_enc_layer(diff)
		diff = self.enc_diff_ebd_layer(diff)

		for convnext_layer in self.enc_layers:
			content_embedding = convnext_layer(content_embedding) 
		output   = self.enc_ebd_layer(content_embedding)

		#----------------DECODER----------------#
		out_list = []
		base = diff
		for layer in self.dec_exc_layers:
			output = layer(output)

		for ii in range(len(self.cfg['kfc_stride'])-1):
			pym = self.dec_shortcuts[3*ii+0](base)
			pym = self.dec_shortcuts[3*ii+1](pym)
			pym = self.dec_shortcuts[3*ii+2](pym)
			output = self.dec_layers[ii]( output )

			memory_z = self.dec_bsm_z[ii](output)
			memory_r = self.dec_bsm_r[ii](pym)
			memory = torch.relu( memory_z + memory_r )
			att    = torch.sigmoid(self.dec_bsm_h[ii](memory))
			output = att*output + (1- att)* memory_r
		
		output = self.dec_layers[ii+1]( output )
		output = self.dec_head_layers(output)
		img_out = torch.sigmoid(output)
		out_list.append(img_out)

		return  out_list


