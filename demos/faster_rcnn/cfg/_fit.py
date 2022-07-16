import torch
import math
import time
import sys

def clip_gradient(model, clip_norm):
	"""Computes a gradient clipping coefficient based on gradient norm."""
	totalnorm = 0
	for p in model.parameters():
		if p.requires_grad:
			modulenorm = p.grad.data.norm()
			totalnorm += modulenorm ** 2
	totalnorm = torch.sqrt(totalnorm).item()
	norm = (clip_norm / max(totalnorm, clip_norm))
	for p in model.parameters():
		if p.requires_grad:
			p.grad.mul_(norm)

def Fit(model, args, optimizer, train_loader, validation_loader):

	for epoch in range(args.start_epoch, args.total_epoch):

		if (epoch + 1) % (8 + 1) == 0:
			for param_group in optimizer.param_groups:
				param_group['lr'] = param_group['lr'] * 0.1

		print('\nEpoch {} learning_rate : {}'.format(epoch+1, optimizer.param_groups[0]['lr']))

		_Train(model, train_loader, optimizer)

		torch.save(model.module.state_dict(), f'./{epoch+1}.pth')

def _Train(model, train_loader, optimizer):

	model.train()
	for batch, (images, targets) in enumerate(train_loader):

		# images = images.cuda(non_blocking=True)
		# targets = targets.cuda(non_blocking=True)

		_, loss_rpn_cls, loss_rpn_box, loss_fast_cls, loss_fast_box = model(images, targets)

		optimizer.zero_grad()

		loss = loss_rpn_cls + loss_rpn_box + loss_fast_cls + loss_fast_box
		loss.backward()

		# for vgg only
		clip_gradient(model, 10.)

		optimizer.step()

		print(loss.item(), loss_rpn_cls.item(), loss_rpn_box.item(), loss_fast_cls.item(), loss_fast_box.item())
