import os
import math
import argparse
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from PIL import Image
from skimage.feature import hog
from skimage import io
from Model import LogisticModel, CNN, SVMTrainer, SVMPredictor, Fisher
from kernel import Kernel
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from MyDataset import MyDataset

train_folder = ['09', '10']
prefixFDDB = 'FDDB-folds/FDDB-fold-'
prefixPics = 'originalPics/'
suffix = '-ellipseList.txt'

pos_path = "samples/positive"
neg_path = "samples/negative"

test_path = 'testdata/'

S = []


def obtainSamples(X, y, path, isPos, model):
	# obtain data
	folds = os.listdir(path)
	for fold in folds:
		try:
			imgfiles = os.listdir(os.path.join(path,fold))
		except:
			print("Not a valid directory!")
			continue

		for imgfile in imgfiles:
			try:
				img = io.imread(os.path.join(path,fold,imgfile))
				print("Processing path: ",os.path.join(path,fold,imgfile))
				if model=="CNN":
					X.append(np.array(img).reshape((3,96,96)))
					if isPos:
						y.append(1)
					else:
						y.append(0)
						
					continue
				# extract hog feature
				hog_feature = hog(img, 
								orientations=9,
								pixels_per_cell=(16, 16),
								cells_per_block=(2,2),
								visualize=False)

				X.append(hog_feature)
				if isPos:
					y.append(1)
				else:
					y.append(0)
			except:
				print("Error")
				continue

	return X, y

def TSNEVisualization(results, labels, title,name):
	print(results.shape)
	tsne = TSNE(n_components=2, perplexity=40, n_iter=300)
	tsne_results = tsne.fit_transform(results)

	color = sns.color_palette("hls", 2)
	# cmap = sns.cubehelix_palette(start = 1.5, rot = 3, gamma=0.8, as_cmap = True)

	fig = plt.figure()
	axes = fig.add_subplot(111)
	legends = {}
	label_legends = np.arange(2)
	for i in range(tsne_results.shape[0]):
		legends[labels[i]] = axes.scatter(tsne_results[i,0],tsne_results[i,1],color=color[labels[i]],alpha=0.5)

	# tmp = axes.scatter(tsne_results[:,0],tsne_results[:,1],c=np.sqrt((np.square(tsne_results[:,1]))+np.square(tsne_results[:,0])),cmap=cmap, alpha=0.75)
	# fig.colorbar(tmp, shrink=0.6)
	plt.legend(legends.values(), label_legends)
	plt.title(title)
	plt.savefig(name)

def plot_contour(X1_train, X2_train, clf):
        # 作training sample数据点的图
        pl.plot(X1_train[:,0], X1_train[:,1], "ro")
        pl.plot(X2_train[:,0], X2_train[:,1], "bo")
        # 做support vectors 的图
        pl.scatter(clf.sv[:,0], clf.sv[:,1], s=100, c="g")
        X1, X2 = np.meshgrid(np.linspace(-6,6,50), np.linspace(-6,6,50))
        X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
        Z = clf.project(X).reshape(X1.shape)
        # pl.contour做等值线图
        pl.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
        pl.contour(X1, X2, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
        pl.contour(X1, X2, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')

        pl.axis("tight")
        pl.show()


def generateclassify(model):
	test_path = 'classify/'
	test_folders = os.listdir(test_path)
	X_test = []
	y_test = []
	for file in test_folders:
		try:
			with Image.open(os.path.join(test_path, file)) as img:
				if model=="CNN":
					X_test.append(np.array(img))
					y_test.append(np.array(img))
				hog_feature = hog(img,
				                  orientations=9,
				                  pixels_per_cell=(16, 16),
				                  cells_per_block=(2, 2),
				                  visualize=False)
				X_test.append(hog_feature)
				y_test.append(0)
		except:
			continue
	return np.array(X_test),np.array(y_test)

def CNNTest(model, X_test, y_test, filename):
	test_dataset = MyDataset(X_test, y_test, transform=transforms.ToTensor())
	test_loader = DataLoader(dataset=test_dataset)
	model.eval()
	accuracy = 0.0
	for i, data in enumerate(test_loader, 1):
		feature, label = data
		with torch.no_grad():
			feature = Variable(feature)
		out, inter_layer = model(feature)
		_, pred = torch.max(out, 1)
		accuracy += (pred != 1).sum().item()
	f = open(filename, 'a')
	f.write("test accuracy: {:.6f}\n".format(accuracy/len(X_test)))
	print("test accuracy: {:.6f}".format(accuracy/len(X_test)))
	f.close()

def SVM_plot(dataArr, labelArr, Support_vector_index, W, b, name):
	for i in range(np.shape(dataArr)[0]):
		if labelArr[i] == 1:
			plt.scatter(dataArr[i][0],dataArr[i][1],c='b',s=20, alpha=0.5)
		else:
			plt.scatter(dataArr[i][0],dataArr[i][1],c='y',s=20, alpha=0.5)
	for j in range(len(Support_vector_index)):
		if Support_vector_index[j]:
			plt.scatter(dataArr[j][0],dataArr[j][1], s=100, c = '', alpha=0.5, linewidth=1.5, edgecolor='red')

	plt.savefig(name+"_SVM.png")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-model', choices=['logistic','CNN','SVM','Fisher'], required=True)
	# This argument is only for logistic regression
	parser.add_argument('-opt', choices=['SGD', 'Langevin'], required=False)
	# This argument is only for CNN
	parser.add_argument('-load', choices=['y','n'], required=False)
	# This argument is only for SVM
	parser.add_argument('-kernel', choices=['linear', 'RBF'], required=False)
	# parser.add_argument('-pattern', choices=['classification','detection'], required=True)
	args = parser.parse_args()

	batch_size = 10
	lr = 0.001
	epochs = 1

	# train data
	X_pos = []
	X_neg = []
	# label
	y_pos = []
	y_neg = []

	# # obtain positive data
	X_pos, y_pos = obtainSamples(X_pos, y_pos, pos_path, True, args.model)

	# obtain negative data
	X_neg, y_neg = obtainSamples(X_neg, y_neg, neg_path, False, args.model)

	X = []
	y = []

	X.extend(X_pos)
	X.extend(X_neg)
	y.extend(y_pos)
	y.extend(y_neg)


	X_test, y_test = generateclassify(args.model)

	# train
	if args.model == 'logistic':
		model = LogisticModel(n_iterations=1000, optimizer=args.opt)
		model.fit(np.array(X), np.array(y))

	if args.model == 'CNN':
		train_dataset = MyDataset(X, y, transform=transforms.ToTensor())
		train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
		model = CNN(3, 2)
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
		filename = "CNNLOG"
		for epoch in range(epochs):
			running_acc = 0.0
			running_loss = 0.0
			visualize = []
			labels = []
			for step, data in enumerate(train_loader, 1):
				feature, label = data
				print("feature:",feature.shape)
				feature = Variable(feature)
				label = Variable(label)

				# forward
				out, inter_layer = model(feature)
				visualize.extend(list(inter_layer.detach().numpy()))
				labels.extend(list(label.detach().numpy()))
				loss = criterion(out, label)

				# backward
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				_, pred = torch.max(out, dim=1)

				# accumulate loss
				running_loss += loss.item()*label.size(0)

				# accumulate the number of correct samples
				current_num = (pred==label).sum()

				acc = (pred == label).float().mean()
				running_acc += current_num.item()

				if step%10 == 0:

					f = open(filename,'a')
					f.write("epoch: {}/{}, loss: {:.6f}, running_acc: {:.6f}\n".format(epoch+1, epochs, loss.item(), acc.item()))
					f.close()
					print("epoch: {}/{}, loss: {:.6f}, running_acc: {:.6f}".format(epoch+1, epochs, loss.item(), acc.item()))
			torch.save(model, "Model/%d"%(epoch+1))
			f = open(filename, 'a')
			f.write("epoch: {}, loss: {:.6f}, accuracy: {:.6f}\n".format(epoch + 1, running_loss,
		                                                         running_acc / len(X)))
			f.close()
			print("epoch: {}, loss: {:.6f}, accuracy: {:.6f}".format(epoch + 1, running_loss,
		                                                         running_acc / len(X)))
			CNNTest(model, X_test, y_test, filename)

		print("drawing figure")
		dim1, dim2 = visualize[0].shape
		length = len(visualize)
		visualize = np.array(visualize).reshape((length*dim1, dim2))
		title1 = "Visualization of the intermediate-layer"
		# title2 = "Visualization of the PCA feature"
		TSNEVisualization(visualize, labels, title1, title1)

	if args.model == 'SVM':
		if args.kernel=='RBF':
			model = SVC(kernel='rbf')
		else:
			model = SVC(kernel='linear')
		model.fit(X,y)
		# 支持向量个数
		n_Support_vector = model.n_support_
		#支持向量索引
		Support_vector_index = model.support_
		# 方向向量W
		if args.kernel=='linear':
			W = model.coef_
			# 截距项b
			b = model.intercept_
			pca = PCA(n_components=2)
			pca_results = pca.fit_transform(X)
			SVM_plot(pca_results,np.array(y),Support_vector_index,W,b,args.kernel+"1")

		# Gkernel = Kernel()
		# if args.kernel=="RBF":
		# 	model = SVMTrainer(Gkernel.gaussian(0.3), 0.5)
		# else:
		# 	model = SVMTrainer(Gkernel.linear(), 0.5)
		# X = np.array(X).astype(float)
		# predictor = model.train(X,np.array(y))
		
		# SVM_plot(pca_results,np.array(y),predictor._support_vector_indices,predictor._weights,predictor._bias,args.kernel)
		# filename = "Support_vector.txt"
		# # np.save("Support_vector", predictor._support_vectors)
		# f = open(filename,'w')
		# for vector in predictor._support_vectors:
		# 	f.write(str(vector)+"\n")
		# 	f.write("------------------------------------------------------------------------------------------------\n")
		# 	f.write("------------------------------------------------------------------------------------------------\n")
		# f.close()


	if args.model == 'Fisher':
		model = Fisher()
		model.fit(np.array(X_pos), np.array(X_neg))


	print("Doing classification---------------------------------------------------")
	accuracy = 0.0
	for hog_feature in X_test:
		if args.model == 'logistic':
			pred = model.predict(hog_feature)
			if pred < 0.1:
				accuracy += 1
		if args.model == 'SVM':
			pred = model.predict([hog_feature])[0]
			if pred < 0.1:
				accuracy += 1
		if args.model == 'Fisher':
			pred = model.predict(hog_feature)
			if pred < 0.1:
				accuracy += 1
	print("accuracy: ", accuracy / len(X_test))

	print("Doing detection---------------------------------------------------------")
	test_path = 'detection/'
	test_folders = os.listdir(test_path)
	accuracy = {}
	total_pic = 0
	for folder in test_folders:
		try:
			test_sample = os.listdir(os.path.join(test_path,folder))
		except:
			print("Not a directory!")
			continue
		face_num = 0
		total_pic += 1
		for path in test_sample:
			if path=="ground_truth.txt":
				with open(os.path.join(test_path, folder, path), 'r') as f:
					face_num = int(f.readlines()[0])
				continue
			print("The size of bounding box: ", path)
			try:
				images = os.listdir(os.path.join(test_path, folder, path))
			except:
				continue
			count = 0
			features = []
			prelabel = []
			for image in images:
				try:
					with Image.open(os.path.join(test_path, folder, path, image)) as img:
						hog_feature = hog(img,
										orientations=9,
										pixels_per_cell=(16, 16),
										cells_per_block=(2,2),
										visualize=False)
						features.append(hog_feature)
						if args.model == 'logistic':
							pred = model.predict(hog_feature)
							if pred>0:
								count += 1
						if args.model == 'SVM':
							pred = model.predict([hog_feature])[0]
							prelabel.append(pred)
							if pred>0:
								count += 1
						if args.model == 'CNN':
							test_dataset = MyDataset(img, transform=transforms.ToTensor())
							test_loader = DataLoader(dataset=test_dataset)
							model.eval()
							for i, data in enumerate(test_loader, 1):
								feature, label = data
								with torch.no_grad():
									feature = Variable(feature)
								out, inter_layer = model(feature)
								_, pred = torch.max(out, 1)
								count += (pred == 1).sum().item()
						if args.model == 'Fisher':
							pred = model.predict(hog_feature)
							if pred>0:
								count += 1
						if pred>0:
							tmppath = "detection_results/"+args.model+"/"+folder+'/'+path+"/"
							if not os.path.exists("detection_results/"+args.model+"/"):
								os.mkdir("detection_results/"+args.model+"/")
							if not os.path.exists("detection_results/"+args.model+"/"+folder+'/'):
								os.mkdir("detection_results/"+args.model+"/"+folder+'/')
							if not os.path.exists(tmppath):
								os.mkdir(tmppath)
							img.save(tmppath+image)
				except:
					continue
			if face_num == count:
				size = int(path)
				if size not in accuracy.keys():
					accuracy[size] = 0.0
				accuracy[size] += 1.0
			print("ground_truth: ", face_num, "prediction number: ", count)
	for key in accuracy.keys():
		print("bounding box size: ", key, " accuracy: ", accuracy[key]/total_pic)
		

if __name__ == '__main__':
	main()



