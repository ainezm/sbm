import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt

class SBM:
	def __init__(self, k, n, a, b):
		self.k = k
		self.n = n
		self.a = float(a)
		self.b = float(b)

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
		edges1 = np.where(edges <= .5)[0]
		edges2 = np.where(edges > .5)[0]

		A = np.zeros((self.n*self.k, self.n*self.k))
		for x, y in zip(ai[edges1],aj[edges1]):
			A[x,y] = 1

		B = np.zeros((self.n*self.k, self.n*self.k))
		for x, y in zip(ai[edges2],aj[edges2]):
			B[x,y] = 1

		# construct adjacency matrix over the random split
		A = A+A.T
		A[A > 0] = 1
		B = B+B.T
		B[B > 0] = 1

		aa = A
		bb = B

		#Spectral partition part
		#Randomly select about half indices in range(k*n)
		r = np.random.rand(self.k*self.n,1)
		Y = np.where(r > .5)[0]
		Z = np.where(r < .5)[0]

		#Randomly select about half of those indices
		r = np.random.rand(len(Y),1)
		Y1 = Y[np.where(r > .5)[0]]
		Y2 = Y[np.where(r < .5)[0]]

		#select half rows and quarter (approx) columns randomly from A
		A1 = A[Z, :][:,Y1]

		#singular value decomposition of A1
		U,S,V = np.linalg.svd(A1)
		
		#get k largest singular values
		max_S = np.argsort(S)[:self.k]
		U = U[:,max_S]
		V = V[max_S,:]

		#Get k equally spaced columns from Y2 - idxs not used in svd
		colsY2 = Y2[np.linspace(0, len(Y2)-1, num=self.k, dtype= np.int16)]
		
		#A2 has same rows as A1, k columns not in A1
		A2 = A[Z,:][:,colsY2] - (self.a+self.b)/(4*self.n)

		#Projection of A2 onto singular values U of A1
		projY2 = np.dot(np.dot(U,U.T),A2)

		#Construct VV - idx is top n/2 coordinates of each proj vector
		VV = np.zeros((self.k,int(self.n/2)))
		for i in range(self.k):
			e1 = projY2[:,i]
			idx = np.argsort(e1)[::-1][:int(self.n/2)]
			VV[i,:] = Z[idx]

		outCxs = self.construct_outCx(VV)
		ns = [len(x) for x in outCxs]

		#find every idx in Z not in any outCx
		extra = Z
		for outCx in outCxs:
			extra = np.setdiff1d(extra, outCx).astype(int)

		#the original adjacency matrix- combining blue and red edges
		A = aa+bb

		dxs = self.construct_dx(A, outCxs, extra)
		idextra = np.argmax(dxs, axis = 0)

		outCxs = self.construct_outCxs_2(outCxs,extra,idextra)

		outCxs = self.correction(outCxs, A)

		#clustering Y
		A = B

		dxs = []

		for outCx in outCxs:
			C1 = A[outCx,:][:,Y]
			dxs.append(np.sum(C1, axis=0))

		idY = np.argmax(dxs, axis = 0)

		for i in range(len(outCxs)):
			outCx = np.union1d(outCxs[i],Y[np.where(idY == i)])
			outCxs[i] = np.random.shuffle(outCx)

		A = aa+bb

		plt.imshow(A[outCxs,outCxs]);
		plt.colorbar()
		plt.show()




	def correction(self, outCxs, A):
		dZZ = 1.5*(self.a+self.b)/4
		bad_xy = {}
		for block1 in range(self.k):
			for block2 in range(block1+1, self.k):
				Cx = A[outCxs[block2],:][:,outCxs[block1]]
				degZ = np.sum(Cx, axis = 0)
				bad12 = outCxs[block1][np.where(degZ>dZZ)[0]]
				bad_xy[(block1,block2)] = bad12
				degZ = np.sum(Cx, axis = 1)
				bad21 = outCxs[block2][np.where(degZ>dZZ)[0]]
				bad_xy[(block2, block1)] = bad21

		newOutCxs = []
		for block1 in range(self.k):
			xy = set()
			yx = set()
			for block2 in range(self.k):
				if block2 == block1:
					continue
				xy = np.union1d(xy, bad_xy[(block1, block2)])
				yx = np.union1d(yx, bad_xy[(block2, block1)])

			outCx = np.setdiff1d(outCxs[block1], xy).astype(int)
			outCx = np.union1d(outCx, yx).astype(int)	
			newOutCxs.append(outCx)
		return newOutCxs




	def construct_outCxs_2(self, outCxs, extra, idextra):
		outCxs = []
		for i in range(len(outCxs)):
			outCx = np.union1d(outCxs[i], extra[np.where(idextra == i)[0]]).astype(int)
			outCxs.append(outCx)
		return outCxs



	def construct_dx(self, A, outCxs, extra):
		dxs = []
		for outCx in outCxs:
			Cx = A[outCx,:][:,extra]
			dx = np.sum(Cx, axis = 0)
			dxs.append(dx)

		return dxs


	def construct_outCx(self, VV):
		outCxs = []
		for block1 in range(self.k):
			outCx = np.setdiff1d(VV[block1,:],VV[np.delete(list(range(self.k)), block1),:]).astype(int)
			outCxs.append(outCx)
		return outCxs


		






if __name__ == "__main__":
	s = SBM(3, 100, 20, 1)
	s.generate_model()







