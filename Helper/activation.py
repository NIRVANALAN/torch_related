import torch


class ReLU(torch.autograd.Function):
	
	@staticmethod
	def forward(ctx, *args, **kwargs):
		ctx.save_for_backward(args[0])
		return args[0].clamp(min=0)
		pass
	
	@staticmethod
	def backward(ctx, *grad_outputs):
		x, = ctx.saved_tensors
		grad_x = grad_outputs[0].clone()
		grad_x[x < 0] = 0
		return grad_x
		pass
