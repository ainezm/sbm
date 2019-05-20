import numpy as np
import scipy.sparse

class SBM:
	def __init__(self, k, n, a, b):
		self.k = k
		self.n = n
		self.a = a
		self.b = b

	def generate_model(self):
		#initialize sparse matrix
		A = np.triu(np.random.rand(self.n*self.k, self.n*self.k))
		A[A > self.b/self.n] = 0

		#create k dense blocks on the diagonal
		for i in range(self.k):
			A11 = np.triu(np.random.rand(self.n))
			A11[A11 > self.a/self.n] = 0
			A[i*self.n:(i+1)*self.n,i*self.n:(i+1)*self.n] = A11

		A = np.triu(A,1)

		#ensure symmetry - construct adjacency matrix
		A = A+A.T
		A[A > 0] = 1

		#find edge idxs
		ai, aj = np.where(np.triu(A)>0)
		m = len(ai)

		#creating a random splitting of edges
		edges = np.random.rand(m,1)
		edges1 = np.where(edges <= .5)
		edges2 = np.where(edges > .5)

		A = np.zeros((self.n*self.k, self.n*self.k))
		for x, y in zip(ai[edges1],aj[edges1]):
			A[x,y] = 1

		B = np.zeros((self.n*self.k, self.n*self.k))
		for x, y in zip(ai[edges2],aj[edges2]):
			B[x,y] = 1

		# construct adjacency matrix
		A = A+A.T
		A[A > 0] = 1
		B = B+B.T
		B[B > 0] = 1

		aa = A
		bb = B




		






if __name__ == "__main__":
	s = SBM(3, 1000, 20, 1)
	s.generate_model()







