import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath


class ConvNeXtBlock(nn.Module):
	def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
		super().__init__()
		self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
		self.norm = LayerNorm(dim, eps=1e-6)
		self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
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

class NeRV_CustomConv(nn.Module):
	def __init__(self, **kargs):
		super(NeRV_CustomConv, self).__init__()

		ngf, new_ngf, kernel, stride, padding = kargs['ngf'], kargs['new_ngf'], kargs['kernel'], kargs['stride'], kargs['padding']
		self.conv_type = kargs['conv_type']
		if self.conv_type == 'conv':
			#self.conv = nn.Conv2d(ngf, new_ngf * stride * stride, 3, 1, 1, bias=kargs['bias'])
			self.conv = nn.Conv2d(ngf, new_ngf * stride * stride, kernel, 1,  padding, bias=kargs['bias'])
			self.up_scale = nn.PixelShuffle(stride)
		elif self.conv_type == 'deconv':
			self.conv = nn.ConvTranspose2d(ngf, new_ngf, stride, stride)
			self.up_scale = nn.Identity()
		elif self.conv_type == 'bilinear':
			self.conv = nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=True)
			self.up_scale = nn.Conv2d(ngf, new_ngf, 2*stride+1, 1, stride, bias=kargs['bias'])

	def forward(self, x):
		out = self.conv(x)
		return self.up_scale(out)

class HNeRVBlock(nn.Module):
	def __init__(self, **kargs):
		super().__init__()

		self.conv = NeRV_CustomConv(ngf=kargs['ngf'], new_ngf=kargs['new_ngf'], kernel=kargs['kernel'], stride=kargs['stride'],
			padding=kargs['padding'], bias=kargs['bias'], conv_type=kargs['conv_type'])
		self.act = nn.GELU()

	def forward(self, x):
		return self.act(self.conv(x))

class D_NeRV(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.c1_dim = 64
		self.d_dim  = 16
		self.c2_dim = 92
		self.r_c = 1.2

		self.cont_enc_list = [5, 4, 4, 3, 2]
		self.diff_enc_list = [4, 3, 2]

		self.cont_dec_stride = [1, 1, 1, 1, 1]
		self.cont_dec_kernel = [1, 3, 5, 5, 5]
		self.cont_dec_padding= [0, 1, 2, 2, 2]

		self.ccu_kernel = [3] #[1]
		self.ccu_stride = [1]
		self.ccu_padding= [1] #[0]

		self.cont_enc_layers, self.cont_dec_layers, self.head_layers = [nn.ModuleList() for _ in range(3)]
		self.diff_enc_layers = nn.ModuleList()
		self.ccu_dec_layers = nn.ModuleList()

		for k, stride in enumerate(self.cont_enc_list):
			if k == 0:
					c0 = 3
			else:
				c0 = self.c1_dim

			self.cont_enc_layers.append(nn.Conv2d(c0, self.c1_dim, kernel_size=self.cont_enc_list[k],
					stride=self.cont_enc_list[k]))
			self.cont_enc_layers.append(LayerNorm(self.c1_dim,t_3d=False, eps=1e-6, data_format="channels_first"))
			self.cont_enc_layers.append(ConvNeXtBlock(dim=self.c1_dim))
			self.enc_embedding_layer = nn.Conv2d(self.c1_dim, self.d_dim, kernel_size=1, stride=1)

			if k<len(self.diff_enc_list):
				if k == 0:
					c0 = 6
				else:
					c0 = self.c1_dim
				self.diff_enc_layers.append(nn.Conv2d(c0, self.c1_dim, kernel_size=self.diff_enc_list[k],
						stride=self.diff_enc_list[k]))
				self.diff_enc_layers.append(LayerNorm(self.c1_dim,t_3d=False, eps=1e-6, data_format="channels_first"))
				self.diff_enc_layers.append(ConvNeXtBlock(dim=self.c1_dim))
			self.diff_enc_ebd_layer = nn.Conv2d(self.c1_dim, 2, kernel_size=1, stride=1)

		self.enc_c2_layer = nn.Conv2d(self.d_dim, self.c2_dim, kernel_size=1, stride=1)
		ngf = self.c2_dim

		for i, stride in enumerate(self.cont_dec_stride):
			new_ngf = round(ngf//self.r_c)

			self.cont_dec_layers.append(HNeRVBlock(ngf=ngf, new_ngf=new_ngf, kernel=self.cont_dec_kernel[i], stride=stride,
				padding=self.cont_dec_padding[i], conv_type='conv'))
			ngf = new_ngf

			head_layer = [None]
			if i == len(self.cont_dec_stride) - 1:
				head_layer = nn.Conv2d(ngf, 3, 3, 1, 1)
			else:
				head_layer = None
			self.head_layers.append(head_layer)

		ngf= int(int(self.c2_dim/1.2)/1.2)
		self.diff_exc_layer = nn.Conv2d(2, ngf, kernel_size=1, stride=1)
		for i, stride in enumerate(self.ccu_stride):
			new_ngf = round(ngf//1.2)
			self.ccu_dec_layers.append(HNeRVBlock(ngf=ngf, new_ngf=new_ngf, kernel=self.ccu_kernel[i],
				stride=stride, padding=self.ccu_padding[i], conv_type='conv'))
			ngf = new_ngf

		ngf_a = int(int(int(self.c2_dim/self.r_c)/self.r_c)/self.r_c)

		self.p_c = nn.Conv2d(ngf_a, ngf_a, kernel_size=3, stride=1, padding=1)
		self.p_d = nn.Conv2d(ngf_a, ngf_a, kernel_size=3, stride=1, padding=1)

		self.s_c = nn.Conv2d(ngf_a, ngf_a, kernel_size=3, stride=1, padding=1)
		self.s_d = nn.Conv2d(ngf_a, ngf_a, kernel_size=3, stride=1, padding=1)

	'''data = {
		"img_id": frame_idx,
		"img_gt": tensor_image,  #3 H W
		"img_p": tensor_image_p, #3 h w
		"img_f": tensor_image_f, #3 h w
	}
	'''
	def forward(self, data):
		#==========================ENCODER==========================#
		#input
		content_embedding = data['img_gt']
		content_p = data['img_p']
		content_f = data['img_f']
		content_gt = data['img_gt']

		#content_encoder
		for encoder_layer in self.cont_enc_layers:
			content_embedding = encoder_layer(content_embedding)
		cnt_output   = self.enc_embedding_layer(content_embedding)
		output   = self.enc_c2_layer(cnt_output)

		#difference_encoder
		diff_p = content_gt - content_p
		diff_f = content_f - content_gt
		diff = torch.stack([diff_p, diff_f], dim=2)
		diff = diff.view(diff.size(0),-1,diff.size(-2),diff.size(-1))
		for diff_enc_layer in self.diff_enc_layers:
			diff = diff_enc_layer(diff)
		diff = self.diff_enc_ebd_layer(diff)
		#==========================ENCODER==========================#

		#==========================DECODER==========================#
		diff = self.diff_exc_layer(diff)
		output = self.cont_dec_layers[0](output)
		output = self.cont_dec_layers[1](output)
		output = self.cont_dec_layers[2](output)

		#-------CCU------#
		diff = self.ccu_dec_layers[0](diff)
		p = torch.tanh(self.p_c(output) + self.p_d(diff))
		s = torch.sigmoid(self.s_c(output) + self.s_d(diff))
		output = s * p + (1-s)*output
		#-------CCU------#

		output = self.cont_dec_layers[3](output)
		output = self.cont_dec_layers[4](output)
		img_out = self.head_layers[-1](output)
		img_out = torch.sigmoid(img_out)
		#==========================DECODER==========================#

		out_list = []
		out_list.append(img_out)

		return  out_list
