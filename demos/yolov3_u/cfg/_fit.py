import torch

import time
import sys

def Fit(model, args, optimizer, criterion, metric, train_loader, validation_loader):

	best_val_metric1 = 0
	best_val_loss = float('inf')
	patient = 0

	for epoch in range(args.start_epoch, args.max_epochs):
		s_time = time.time()

		print('\nEpoch {} learning_rate : {}'.format(epoch+1, optimizer.param_groups[0]['lr']))

		loss_train = _Train(model, train_loader, optimizer, criterion, epoch)
		loss_val = _Validate(model, validation_loader, criterion, epoch)

		if loss_val < best_val_loss:
			torch.save(model.state_dict(), f'./epoch_{epoch+1}_loss_{loss_val}.pth')
			best_val_loss = loss_val
			patient = 0

		if patient >= 3:
			for param_group in optimizer.param_groups:
				param_group['lr'] = param_group['lr'] * 0.1 if param_group['lr'] * 0.1 > 1e-8 else 1e-8
			patient = 0

		patient += 1

		print('epoch : {} train_loss : {:.3f} time : {:.3f}'.format(epoch+1, loss_train, time.time() - s_time))
		print('epoch : {} val_loss : {:.3f} time : {:.3f}'.format(epoch+1, loss_val, time.time() - s_time))


def _Train(model, train_loader, optimizer, criterion, epoch):

	model.train()

	loss_batch = []
	for batch, (images, target) in enumerate(train_loader):

		s_time = time.time()

		images = images.cuda(non_blocking=True)
		target = target.cuda(non_blocking=True)

		predict = model(images)

		optimizer.zero_grad()
		loss = criterion(predict, target, model)
		loss.backward()
		optimizer.step()

		loss_batch.append(loss.item())
		print('epoch : {} batch : {} / {} loss : {:.3f} time : {:.3f}'.format(epoch+1, batch+1, len(train_loader), loss.item(), time.time() - s_time))

	return sum(loss_batch) / len(loss_batch)

def _Validate(model, val_loader, criterion, epoch):
	model.eval()

	loss_batch = []
	with torch.no_grad():
		for batch, (images, target) in enumerate(val_loader):
			s_time = time.time()

			images = images.cuda(non_blocking=True)
			target = target.cuda(non_blocking=True)

			predict = model(images)

			loss = criterion(predict, target, model)

			loss_batch.append(loss.item())
			print('epoch : {} batch : {} / {} loss : {:.3f} time : {:.3f}'.format(epoch+1, batch+1, len(val_loader), loss.item(), time.time() - s_time))

	return sum(loss_batch) / len(loss_batch)
