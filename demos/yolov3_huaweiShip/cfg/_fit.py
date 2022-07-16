import torch
import math
import time
import sys

def Fit(model, args, optimizer, criterion, metric, scheduler, train_loader, validation_loader):

	best_val_metric1 = 0
	best_loss = float('inf')
	patient = 0

	for epoch in range(args.start_epoch, args.total_epoch):
		s_time = time.time()

		print('\nEpoch {} learning_rate : {}'.format(epoch+1, optimizer.param_groups[0]['lr']))

		if epoch < args.total_epoch - args.no_aug_epoch:
			loss_train = _Train(model, train_loader, optimizer, criterion, epoch, args)
		else:
			print('using validation_loader for no_aug')
			loss_train = _Train(model, validation_loader, optimizer, criterion, epoch, args)

		if loss_train < best_loss:
			torch.save(model.module.state_dict(), f'./best-epoch_{str(epoch + 1).rjust(3, "0")}_loss_{loss_train}.pth')
			best_loss = loss_train

		torch.save(model.module.state_dict(), f'./epoch_{str(epoch + 1).rjust(3, "0")}_loss_{loss_train}.pth')

		print('epoch : {} train_loss : {:.3f} time : {:.3f}'.format(str(epoch + 1).rjust(3, '0'), loss_train, time.time() - s_time))

		if (epoch >= args.warmup_epoch) and (epoch < args.total_epoch - args.no_aug_epoch - 1):
			scheduler.step()

def _Train(model, train_loader, optimizer, criterion, epoch, args):

	model.train()
	loss_batch = []

	batchs = len(train_loader)
	for batch, (images, target) in enumerate(train_loader):
		s_time = time.time()

		if epoch < args.warmup_epoch:
			cur_iter = epoch * batchs + batch
			lr = (args.init_lr - args.warmup_init_lr) * cur_iter / float(args.warmup_iters) + args.warmup_init_lr
			for param_group in optimizer.param_groups:
				param_group['lr'] = lr
		if epoch >= args.total_epoch - args.no_aug_epoch:
			for param_group in optimizer.param_groups:
				param_group['lr'] = args.min_lr


		images = images.cuda(non_blocking=True)
		target = target.cuda(non_blocking=True)

		predict = model(images)

		optimizer.zero_grad()
		loss_box, loss_cls, loss_conf = criterion(predict, target, model)
		loss_record = loss_box + loss_cls + loss_conf

		# loss_xy *= 10
		# loss_wh *= 0.05
		# loss_cls *= 0.5
		# loss_conf *= 1.0
		loss = loss_box + loss_cls + loss_conf

		loss.backward()
		optimizer.step()

		loss_batch.append(loss.item())
		print('epoch : {} batch : {} / {} loss : {:.3f} lr : {:.8f} time : {:.3f}'.format(str(epoch + 1).rjust(3, "0"), batch+1, len(train_loader), loss_record.item(), optimizer.param_groups[0]['lr'], time.time() - s_time))

	return sum(loss_batch) / len(loss_batch)
