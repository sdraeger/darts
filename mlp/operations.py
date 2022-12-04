import torch.nn as nn


OPS = {
	'none': lambda in_f, out_f: Zero(),
	'dense': lambda in_f, out_f: nn.Linear(in_f, out_f),
	'relu': lambda in_f, out_f: nn.ReLU(),
	'sigmoid': lambda in_f, out_f: nn.Sigmoid(),
	'tanh': lambda in_f, out_f: nn.Tanh(),
	'skip_connect': lambda in_f, out_f: Identity()
}


class Identity(nn.Module):
	"""Simply forwards its input."""

	def __init__(self):
		super(Identity, self).__init__()

	def forward(self, x):
		return x


class Zero(nn.Module):
	"""The paper's 'zero' operation. This is simply a missing link between cells, i.e. multiplication by zero."""
	
	def __init__(self):
		super(Zero, self).__init__()

	def forward(self, x):
		return x.mul(0.0)
