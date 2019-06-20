import os
import numpy as np
import tensorflow as tf
import torch
from torch import nn, optim
import torch.nn.functional as F
import cvxopt.solvers

class LogisticModel(object):
	def __init__(self, learning_rate=0.01, n_iterations=200, optimizer="SGD"):
		self.lr = learning_rate
		self.n_iterations = n_iterations
		self.optimizer = optimizer
		self.weights = None
		self.train_data = None
		self.train_label = None
		self.test_data = None

	def __sigmoid__(self, x):
		return 1.0/(1+np.exp(-x))

	def saveModel(self, Modelname):
		model_path = "Model/"+Modelname
		if not os.path.isdir("Model"):
			os.mkdir("Model")
		np.savetxt(model_path, self.weights)

	def loadModel(self, Modelname):
		model_path = "Model/"+Modelname
		if not os.path.isfile(model_path):
			print("There is no such a file.")
		self.weights = np.loadtxt(model_path)
		self.weights = self.weights.reshape((self.weights.size, 1))

	def fit(self, X_train, Y_train):
		self.train_data = X_train
		self.train_label = Y_train.copy()
		# the label of logistic regression: {0, 1}
		self.train_label[self.train_label<0] = 0

		num_sample, num_feature = self.train_data.shape
		self.weights = np.ones((num_feature, 1))
		print(self.weights.shape)

		for k in range(self.n_iterations):
			# print(self.weights)
			for i in range(num_sample):
				out = self.__sigmoid__(np.matmul(self.train_data[i, :].reshape((1, num_feature)),self.weights))
				error = self.train_label[i]-out
				gradient = self.train_data[i, :].reshape((num_feature, 1))*error[0][0]
				if self.optimizer == 'SGD':
					self.weights += (self.lr*gradient)
				if self.optimizer == "Langevin":
					noise = np.random.normal(size=gradient.shape,scale=np.sqrt(self.lr))
					self.weights += (0.5*gradient*self.lr+noise)
			print("n_iter: {}    error: {}".format(k+1, error[0][0]))

	def predict(self, X_test):
		self.test_data = X_test
		num_feature = self.test_data.size
		dim_weights = self.weights.size

		result = self.__sigmoid__(np.matmul(self.test_data.reshape(1, dim_weights), self.weights))

		if result>0.5:
			pred = 1
		else:
			pred = 0
		return pred


class Fisher(object):
	"""docstring for Fisher"""
	def __init__(self):
		super(Fisher, self).__init__()
		self.weight = None
		self.u1 = None
		self.u2 = None

	def cal_cov_avg(self, samples):
		# compute covariance and average
		cov = np.cov(samples,rowvar=False)
		u = np.mean(samples, axis=0)
		return cov, u

	def fit(self, c1, c2):
		cov1, self.u1 = self.cal_cov_avg(c1)
		cov2, self.u2 = self.cal_cov_avg(c2)
		s_w = len(c1)*cov1+len(c2)*cov2
		self.weight = np.matmul(np.linalg.inv(s_w),self.u1-self.u2)

	def predict(self, X_test):
		center1 = np.dot(self.weight.T, self.u1)
		center2 = np.dot(self.weight.T, self.u2)
		pos = np.dot(self.weight.T, X_test)
		return abs(pos-center1) < abs(pos-center2)


class CNN(nn.Module):
	def __init__(self, in_dim,out_dim):
		super(CNN,self).__init__()

		self.conv1 = nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
			nn.ReLU(inplace=True)
		)
		self.fc1 = nn.Sequential(
			nn.Linear(in_features=32*94*94, out_features=1024),
			nn.ReLU(inplace=True),
			nn.Dropout()
		)
		self.fc2 = nn.Sequential(
			nn.Linear(in_features=1024, out_features=512),
			nn.ReLU(inplace=True),
			nn.Dropout()
		)
		self.fc3 = nn.Linear(in_features=512, out_features=out_dim)

	def forward(self,x):
		
		x = x.float()
		x = self.conv1(x)
		print(x.shape)
		inter_layer = x.view(-1, 32*94*94)
		y = self.fc3(self.fc2(self.fc1(inter_layer)))
		return y, inter_layer
		
MIN_SUPPORT_VECTOR_MULTIPLIER = 1e-5

class SVMTrainer(object):
	def __init__(self, kernel, c):
		self._kernel = kernel
		self._c = c

	def train(self, X, y):
		"""Given the training features X with labels y, returns a SVM
		predictor representing the trained SVM.
		"""
		lagrange_multipliers = self._compute_multipliers(X, y)
		return self._construct_predictor(X, y, lagrange_multipliers)

	def _gram_matrix(self, X):
		n_samples, n_features = X.shape
		K = np.zeros((n_samples, n_samples))
		# TODO(tulloch) - vectorize
		for i, x_i in enumerate(X):
			for j, x_j in enumerate(X):
				K[i, j] = self._kernel(x_i, x_j)
				return K

	def _construct_predictor(self, X, y, lagrange_multipliers):
		support_vector_indices = \
			lagrange_multipliers > MIN_SUPPORT_VECTOR_MULTIPLIER

		support_multipliers = lagrange_multipliers[support_vector_indices]
		support_vectors = X[support_vector_indices]
		support_vector_labels = y[support_vector_indices]

		# bias = y_k - \sum z_i y_i  K(x_k, x_i)
		# Thus we can just predict an example with bias of zero, and
		# compute error.
		bias = np.mean(
			[y_k - SVMPredictor(
				kernel=self._kernel,
				bias=0.0,
				weights=support_multipliers,
				support_vectors=support_vectors,
				support_vector_indices=support_vector_indices,
				support_vector_labels=support_vector_labels).predict(x_k)
			 for (y_k, x_k) in zip(support_vector_labels, support_vectors)])

		return SVMPredictor(
			kernel=self._kernel,
			bias=bias,
			weights=support_multipliers,
			support_vectors=support_vectors,
			support_vector_labels=support_vector_labels,
			support_vector_indices=support_vector_indices)

	def _compute_multipliers(self, X, y):
		n_samples, n_features = X.shape

		K = self._gram_matrix(X)
		# Solves
		# min 1/2 x^T P x + q^T x
		# s.t.
		#  Gx \coneleq h
		#  Ax = b

		P = cvxopt.matrix(np.outer(y, y) * K)
		q = cvxopt.matrix(-1 * np.ones(n_samples))

		# -a_i \leq 0
		# TODO(tulloch) - modify G, h so that we have a soft-margin classifier
		G_std = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
		h_std = cvxopt.matrix(np.zeros(n_samples))

		# a_i \leq c
		G_slack = cvxopt.matrix(np.diag(np.ones(n_samples)))
		h_slack = cvxopt.matrix(np.ones(n_samples) * self._c)

		G = cvxopt.matrix(np.vstack((G_std, G_slack)))
		h = cvxopt.matrix(np.vstack((h_std, h_slack)))

		y = y.astype(float)

		A = cvxopt.matrix(y, (1, n_samples))
		b = cvxopt.matrix(0.0)

		solution = cvxopt.solvers.qp(P, q, G, h, A, b)

		# Lagrange multipliers
		return np.ravel(solution['x'])


class SVMPredictor(object):
	def __init__(self,
				 kernel,
				 bias,
				 weights,
				 support_vectors,
				 support_vector_labels,
				 support_vector_indices):
		self._kernel = kernel
		self._bias = bias
		self._weights = weights
		self._support_vectors = support_vectors
		self._support_vector_labels = support_vector_labels
		self._support_vector_indices = support_vector_indices
		assert len(support_vectors) == len(support_vector_labels)
		assert len(weights) == len(support_vector_labels)
		# print("Bias: %s", self._bias)
		# print("Weights: %s", self._weights)
		# print("Support vectors: %s", self._support_vectors)
		# print("Support vector labels: %s", self._support_vector_labels)

	def predict(self, x):
		"""
		Computes the SVM prediction on the given features x.
		"""
		result = self._bias
		for z_i, x_i, y_i in zip(self._weights,
								 self._support_vectors,
								 self._support_vector_labels):
			result += z_i * y_i * self._kernel(x_i, x)
		return np.sign(result).item()

class SVM(object):

    def __init__(self, kernel, C=None):
        self.kernel = kernel
        self.C = C
        if self.C is not None: self.C = float(self.C)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])

        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1,n_samples))
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        # Lagrange multipliers
        '''
         数组的flatten和ravel方法将数组变为一个一维向量（铺平数组）。
         flatten方法总是返回一个拷贝后的副本，
         而ravel方法只有当有必要时才返回一个拷贝后的副本（所以该方法要快得多，尤其是在大数组上进行操作时）
       '''
        a = np.ravel(solution['x'])
        # Support vectors have non zero lagrange multipliers
        '''
        这里a>1e-5就将其视为非零
        '''
        sv = a > 1e-5     # return a list with bool values
        ind = np.arange(len(a))[sv]  # sv's index
        self.a = a[sv]
        self.sv = X[sv]  # sv's data
        self.sv_y = y[sv]  # sv's labels
        print("%d support vectors out of %d points" % (len(self.a), n_samples))

        # Intercept
        '''
        这里相当于对所有的支持向量求得的b取平均值
        '''
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
        self.b /= len(self.a)

        # Weight vector
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                # linear_kernel相当于在原空间，故计算w不用映射到feature space
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

    def project(self, X):
        # w有值，即kernel function 是 linear_kernel，直接计算即可
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        # w is None --> 不是linear_kernel,w要重新计算
        # 这里没有去计算新的w（非线性情况不用计算w）,直接用kernel matrix计算预测结果
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
            return y_predict + self.b

    def predict(self, X):
        return np.sign(self.project(X))
