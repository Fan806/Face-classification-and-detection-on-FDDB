import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
	def __init__(self, hog_feature, label=None, transform=None):
		self.hog_feature = hog_feature
		self.label = label

	def __getitem__(self, index):
		feature = self.hog_feature[index]
		label = self.label[index]
		return feature, label

	def __len__(self):
		return len(self.hog_feature)