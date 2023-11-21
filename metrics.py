import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import numpy as np


class tfAdaCos(nn.Module):
	"""
	PyTorch implementation of AdaCos layer. Reference: https://arxiv.org/abs/1905.00292
	forked https://www.kaggle.com/code/chankhavu/keras-layers-arcface-cosface-adacos
	Arguments:
		num_features: number of embedding features
		num_classes: number of classes to classify
		is_dynamic: if False, use Fixed AdaCos. Else, use Dynamic Adacos.
	"""
	def __init__(self, num_features, num_classes, is_dynamic=True, m=28.6):
		super(tfAdaCos, self).__init__()
		self.num_features = num_features
		self.num_classes = num_classes
		self.init_s = math.sqrt(2) * math.log(num_classes - 1)
		self.is_dynamic = is_dynamic
		self.m = m

		self.w = nn.Parameter(torch.FloatTensor(num_classes, num_features))
		nn.init.xavier_uniform_(self.w)

#		if self.is_dynamic:
		self.s = nn.Parameter(torch.Tensor(1))
		nn.init.constant_(self.s, self.init_s)
		#	self.s.requires_grad = False

	def forward(self, embedding, label):

		x = F.normalize(embedding, p=2, dim=-1)
		w = F.normalize(self.w, p=2, dim=-1)
		logits = x @ w.T

		#fixed AdaCos
		if not self.is_dynamic:
			return self.init_s * logits + self.m

		#no label (validation)
		if label is None:
			return logits

		label = label.view(-1)

		#dynamic AdaCos
		with torch.no_grad():

			eps = torch.finfo(logits.dtype).eps
			theta = torch.acos(torch.clamp(logits, -1.0+eps, 1.0-eps))
			one_hot = F.one_hot(label, num_classes=self.num_classes).to(logits.dtype)
			b_avg = torch.where(one_hot < 1.0, torch.exp(self.s * logits), torch.zeros_like(logits))
			b_avg = torch.mean(torch.sum(b_avg, dim=1))
			theta_med = torch.median(theta[one_hot == 1])

			self.s.data.copy_(torch.log(b_avg) / torch.cos(torch.min(math.pi/4 * torch.ones_like(theta_med), theta_med)))

		return self.s * logits
