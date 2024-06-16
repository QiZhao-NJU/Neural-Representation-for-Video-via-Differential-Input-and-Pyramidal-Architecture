# ---- Differnece NeRV ---- #
# ---- QiZhao CVPR2023 ---- #
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_utils import ActivationLayer
from timm.models.layers import trunc_normal_, DropPath


class ConvNeXtBlock(nn.Module):
	def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
		super().__init__()
		self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
		self.norm = LayerNorm(dim, eps=1e-6)
		self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1, stride=1)

		self.act = nn.GELU()
		self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1, stride=1)
		self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
		self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

	def forward(self, x):
		input = x
		x = self.dwconv(x)
		x = self.pwconv1(x)
		x = self.act(x)
		x = self.pwconv2(x)
		x = input + self.drop_path(x)
		return x

class LayerNorm(nn.Module):
	def __init__(self, normalized_shape, t_3d=False, eps=1e-6, data_format="channels_last"):
		super().__init__()
		self.weight = nn.Parameter(torch.ones(normalized_shape))
		self.bias = nn.Parameter(torch.zeros(normalized_shape))
		self.t_3d = t_3d
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
			if self.t_3d == True:
				x = self.weight[:,None , None, None] * x + self.bias[:,None, None, None]
			else:
				x = self.weight[:, None, None] * x + self.bias[:, None, None]
			return x

class NeRV_CustomConv(nn.Module):
	def __init__(self, **kargs):
		super(NeRV_CustomConv, self).__init__()

		ngf, new_ngf, kernel, stride, padding = kargs['ngf'], kargs['new_ngf'], kargs['kernel'], kargs['stride'], kargs['padding']
		self.conv_type = kargs['conv_type']
		if self.conv_type == 'conv':
			self.conv = nn.Conv2d(ngf, new_ngf * stride * stride, kernel, 1,  padding)
			self.up_scale = nn.PixelShuffle(stride)
		elif self.conv_type == 'deconv':
			self.conv = nn.ConvTranspose2d(ngf, new_ngf, stride, stride)
			self.up_scale = nn.Identity()
		elif self.conv_type == 'bilinear':
			self.conv = nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=True)
			self.up_scale = nn.Conv2d(ngf, new_ngf, 2*stride+1, 1, stride)

	def forward(self, x):
		out = self.conv(x)
		return self.up_scale(out)

class HNeRVBlock(nn.Module):
	def __init__(self, **kargs):
		super().__init__()
		self.conv = NeRV_CustomConv(ngf=kargs['ngf'], new_ngf=kargs['new_ngf'], kernel=kargs['kernel'], stride=kargs['stride'],
			padding=kargs['padding'], conv_type=kargs['conv_type'])
		self.act = ActivationLayer(kargs['act'])

	def forward(self, x):
		return self.act(self.conv(x))


class D_NeRV_Generator(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.c1_dim = cfg['c1'] #64
		self.d_dim  = cfg['d'] #16
		self.c2_dim = cfg['c2'] #32
		self.r_c    = 1.2
		self.c_h    = int(cfg['height']/(5*4*3*2*2))
		self.c_w    = int(cfg['width']/(5*4*3*2*2))

		self.encoder_layers, self.decoder_layers, self.dec_head_layers = [nn.ModuleList() for _ in range(3)]
		self.diff_enc_layers = nn.ModuleList()
		self.diff_dec_layers = nn.ModuleList()

		for k, stride in enumerate(cfg['encoder_list']):
			if k == 0:
					c0 = 3
			else:
				c0 = self.c1_dim

			self.encoder_layers.append(nn.Conv2d(c0, self.c1_dim, kernel_size=cfg['encoder_list'][k], stride=cfg['encoder_list'][k]))
			self.encoder_layers.append(LayerNorm(self.c1_dim,t_3d=False, eps=1e-6, data_format="channels_first"))
			self.encoder_layers.append(ConvNeXtBlock(dim=self.c1_dim))
			self.enc_embedding_layer = nn.Conv2d(self.c1_dim, self.d_dim, kernel_size=1, stride=1)

			if k<len(cfg['diff_enc_list']):
				if k == 0:
					c0 = 6
				else:
					c0 = self.c1_dim
				self.diff_enc_layers.append(nn.Conv2d(c0, self.c1_dim, kernel_size=cfg['diff_enc_list'][k],
						stride=cfg['diff_enc_list'][k]))
				self.diff_enc_layers.append(LayerNorm(self.c1_dim,t_3d=False, eps=1e-6, data_format="channels_first"))
				self.diff_enc_layers.append(ConvNeXtBlock(dim=self.c1_dim))

			self.diff_enc_ebd_layer = nn.Conv2d(self.c1_dim, 2, kernel_size=1, stride=1)
		self.enc_c2_layer = nn.Conv2d(self.d_dim, self.c2_dim, kernel_size=1, stride=1)
		ngf = self.c2_dim

		for i, stride in enumerate(cfg['stride_list']):
			new_ngf = round(ngf//self.r_c)
			self.decoder_layers.append(HNeRVBlock(ngf=ngf, new_ngf=new_ngf, kernel=cfg['kernel_list'][i], stride=stride,
				padding=cfg['padding_list'][i], act=cfg['act'], conv_type=cfg['conv_type']))
			ngf = new_ngf

			head_layer = [None]
			if i == len(cfg['stride_list']) - 1:
				head_layer = nn.Conv2d(ngf, 3, 3, 1, 1)
			else:
				head_layer = None
			self.dec_head_layers.append(head_layer)

		ngf= int(int(self.c2_dim/1.2)/1.2)
		self.diff_exc_layer = nn.Conv2d(2, ngf, kernel_size=1, stride=1)
		for i, stride in enumerate(cfg['diff_dec_stride']):
			new_ngf = round(ngf//1.2)
			self.diff_dec_layers.append(HNeRVBlock(ngf=ngf, new_ngf=new_ngf, kernel=cfg['diff_dec_kernel'][i],
				stride=stride, padding=cfg['diff_dec_padding'][i], act=cfg['act'], conv_type=cfg['conv_type']))
			ngf = new_ngf

		ngf_a = int(int(int(self.c2_dim/self.r_c)/self.r_c)/self.r_c)
		ngf_a = int(int(int(self.c2_dim/self.r_c)/self.r_c)/self.r_c)
		self.dec_p_c = nn.Conv2d(ngf_a, ngf_a, kernel_size=3, stride=1, padding=1)
		self.dec_p_d = nn.Conv2d(ngf_a, ngf_a, kernel_size=3, stride=1, padding=1)

		self.dec_s_c = nn.Conv2d(ngf_a, ngf_a, kernel_size=3, stride=1, padding=1)
		self.dec_s_d = nn.Conv2d(ngf_a, ngf_a, kernel_size=3, stride=1, padding=1)


	def forward(self, data):
		'''data:
			"img_id": frame_idx,
			"img_gt": tensor_image, C H W
			"img_p": tensor_image_p, #3 h w
			"img_f": tensor_image_f, #3 h w
		'''
		content_embedding = data['img_gt']
		content_p = data['img_p']
		content_f = data['img_f']
		content_gt = data['img_gt']

		# ---------- encoder ---------- #
		for encoder_layer in self.encoder_layers:
			content_embedding = encoder_layer(content_embedding)
		cnt_output   = self.enc_embedding_layer(content_embedding)

		diff_p = content_gt - content_p
		diff_f = content_f - content_gt
		diff =  torch.stack([diff_p, diff_f], dim=2)
		diff = diff.view(diff.size(0),-1,diff.size(-2),diff.size(-1))
		for diff_enc_layer in self.diff_enc_layers:
			diff = diff_enc_layer(diff)
		diff = self.diff_enc_ebd_layer(diff)
		output   = self.enc_c2_layer(cnt_output)

		# ---------- decoder ---------- #
		out_list = []
		diff = self.diff_exc_layer(diff)
		for n in range(2):
			output = self.decoder_layers[n](output)
		output = self.decoder_layers[2](output)
		diff = self.diff_dec_layers[0](diff)

		p = torch.tanh(self.dec_p_c(output) + self.dec_p_d(diff))
		s = torch.sigmoid(self.dec_s_c(output) + self.dec_s_d(diff))
		output = s * p + (1-s)*output

		output = self.decoder_layers[3](output)
		output = self.decoder_layers[4](output)
		img_out = self.dec_head_layers[-1](output)
		img_out = torch.sigmoid(img_out)
		out_list.append(img_out)

		return out_list
