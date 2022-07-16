import numpy as np
from matplotlib import pyplot as plt
import os
import tqdm
import multiprocessing

from .iou import wh_iou_batch


class KMeans():
	def __init__(self, xs, k=9):
		self.num_samples = len(xs)
		self.samples = xs
		self.k = k

		np.random.shuffle(self.samples)

		self.centers = self.samples[:k, :]

	def cal_distance(self, xs, centers):
		# 1 - iou(x, center)
		iou = wh_iou_batch(xs, centers)
		return 1 - iou

	def fit(self, iters):
		for i in range(iters):
			self._fit()

		return self.centers, self.categories

	def _fit(self):
		distance = self.cal_distance(self.samples, self.centers)
		self.categories = np.argmin(distance, axis=1) + 1

		new_centers = []
		for category_id in range(1, self.k+1):
			sample_category = self.samples[self.categories==category_id]

			if sample_category.shape[0] == 0:
				new_centers.append(self.centers[category_id-1, :])
			else:
				new_x = np.mean(sample_category[:, 0])
				new_y = np.mean(sample_category[:, 1])
				new_centers.append([new_x, new_y])
		self.centers = np.array(new_centers).reshape([-1, 2])

def _load_data(labels, wh_normal):
	wh_normal.append(labels[:, 4:].cpu().numpy())

class AnchorGenerator():
	def __init__(self, data_loaders:list, k=9, iters=100, plot=True, save_dir='./'):

		self.data_loaders = data_loaders
		self.k = k
		self.iters = iters

		self.save_dir = save_dir
		self.plot = plot

	def load_data(self):

		wh_normal = []
		for i, loader in enumerate(self.data_loaders):
			for (images, labels) in tqdm.tqdm(loader, desc=f'Anchor Generator extract width-height from {i+1}-th dataloader '):
				self.input_height, self.input_widht = images.size()[2:]
				wh_normal.append(labels[:, 4:].cpu().numpy())
		wh_normal = np.concatenate(wh_normal, axis=0)

		return wh_normal

	def load_cache(self):
		with open(self.cache, 'r') as f:
			centers = eval(f.read())
		return centers

	def get_anchors(self):

		wh_normal = self.load_data()
		wh_normal = np.array(wh_normal, dtype=np.float32).reshape([-1, 2])

		centers, categories = KMeans(xs=wh_normal, k=self.k).fit(iters=self.iters)
		centers = centers.tolist()
		centers.sort(key=lambda x: -x[0]*x[1])
		centers= np.array(centers, dtype=np.float).reshape([-1, 2])

		if self.plot:
			for k in range(1, self.k+1):
				plt.scatter(wh_normal[categories==k, 0], wh_normal[categories==k, 1], c=np.zeros_like(wh_normal[categories==k, 0]).fill(k), alpha=0.8)
				plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='x')
			plt.savefig(f'{self.save_dir}/anchor.png')

		centers = centers * np.array([self.input_widht, self.input_height])
		with open(os.path.join(self.save_dir, 'anchor.txt'), 'w') as f:
			f.write(str(centers.tolist()))

		return centers