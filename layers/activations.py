import torch
import torch.nn as nn
import torch.nn.functional as F

class SILU(nn.Module):

	def __init__(self):
		super(SILU, self).__init__()

		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		return x * self.sigmoid(x)