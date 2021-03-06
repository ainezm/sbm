import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
import itertools
import time

class SBM:
    def __init__(self, k, n, a, b):
        self.k = k
        self.n = n
        self.a = float(a)
        self.b = float(b)

    def run(self):
        start = time.time(); beginning_of_time = start
        A = self.generate_model()
        print("generate SBM takes: %s sec" %(time.time() - start))
        start = time.time()
        reordering = np.random.permutation(self.n*self.k)
        A_shuffled = A[reordering,:][:,reordering]
        print("shuffle takes: %s sec" %(time.time() - start))
        start = time.time()
        red,blue = self.split_red_blue_edges(A_shuffled)
        print("split red blue edges takes: %s sec" %(time.time() - start))
        start = time.time()
        (Y,Z) = self.create_bipartite_graph()
        print("create bipartite graph: %s sec" %(time.time() - start))
        start = time.time()
        (outCxs, Z, Y) = self.spectral_partition(red, blue, Y, Z)
        print("spectral_partition takes: %s sec" %(time.time() - start))
        start = time.time()
        [print(outCx.shape) for outCx in outCxs]
        print(Z.shape)
        outCxs = self.correction(outCxs, A_shuffled, Z)
        print("correction takes: %s sec" %(time.time() - start))
        start = time.time()
        recovered = self.merge(outCxs, A_shuffled, Y, blue)
        print("merge takes: %s sec" %(time.time() - start))
        print("total time: %s sec" % (time.time() - beginning_of_time))
        self.plot_output(A, A_shuffled, recovered)

    def generate_model(self):
        # initialize sparse matrix
        start = time.time()
        A = np.triu(np.random.rand(self.n*self.k, self.n*self.k))
        A[A > self.b/self.n] = 0
        print("create triu matrix %s sec" % (time.time() - start))
        start = time.time()
        #create k dense blocks on the diagonal
        for i in range(self.k):
            A11 = np.triu(np.random.rand(self.n, self.n))
            A11[A11 > self.a/self.n] = 0
            A[i*self.n:(i+1)*self.n,i*self.n:(i+1)*self.n] = A11
        print("create k block %s sec" % (time.time() - start))
        start = time.time()
        # random_values = np.random.rand((self.n*self.k)**2)
        # A = sparse.lil_matrix((self.n*self.k, self.n*self.k))
        # l = 0
        # for k in range(self.k):
        #     for i in range(k*self.n, (k+1)*self.n):
        #         for j in range(i,A.shape[1]):
        #             if k*self.n <= j <= (k+1)*self.n:
        #                 if random_values[l] <= self.a/self.n:
        #                     A[i,j] = 1
        #             else:
        #                 if random_values[l] <= self.b/self.n:
        #                     A[i,j] = 1
        #             l += 1
        # start = time.time()
        # A = sparse.triu(sparse.random(self.n*self.k, self.n*self.k, density=self.b/self.n, data_rvs=lambda x: np.ones(x)))
        # print("create sparse b/n density matrix %s sec" % (time.time() - start))
        # start = time.time()
        # random_values = np.random.rand((self.n)**2)
        # print("create nxn random values %s sec" % (time.time() - start))
        # start = time.time()
        # A = A.tolil()
        # #create k dense blocks on the diagonal
        # block_threshold = self.a/self.n
        # for k in range(self.k):
        #     l = 0
        #     for i in range(k*self.n, (k+1)*self.n):
        #         for j in range(k*self.n, (k+1)*self.n):
        #             A[i,j] = 0
        #             if random_values[l] < block_threshold:
        #                 A[i,j] = 1
        #             l += 1
        #     np.random.shuffle(random_values)
        # print("create a/n density block inside A %s sec" % (time.time() - start))
        # start = time.time()
        A = np.triu(A,1)
        print("create triu 1 %s sec" % (time.time() - start))
        start = time.time()
        #ensure symmetry - construct adjacency matrix
        A = sparse.csr_matrix(A)+sparse.csr_matrix(A.T)
        print("create adjacency matrix %s sec" % (time.time() - start))
        start = time.time()
        # A[A > 0] = 1
        A = sparse.csr_matrix(A)
        print("clean up %s sec" % (time.time() - start))
        return A

    def split_red_blue_edges(self, A):
        #find edge idxs
        (ai, aj, _) = sparse.find(sparse.triu(A)>0)
        m = len(ai)
        #creating a random splitting of edges
        edges = np.random.rand(m,1)
        edges1 = np.where(edges <= .5)[0]
        edges2 = np.where(edges > .5)[0]

        A = sparse.lil_matrix((self.n*self.k, self.n*self.k))
        for x, y in zip(ai[edges1],aj[edges1]):
            A[x,y] = 1

        B = sparse.lil_matrix((self.n*self.k, self.n*self.k))
        for x, y in zip(ai[edges2],aj[edges2]):
            B[x,y] = 1

        # construct adjacency matrix over the random split
        A = A+A.T
        B = B+B.T
        return sparse.csr_matrix(A), sparse.csr_matrix(B)

    def create_bipartite_graph(self):
        #Randomly select about half indices in range(k*n)
        r = np.random.rand(self.k*self.n,1)
        Y = np.where(r > .5)[0]
        Z = np.where(r <= .5)[0]
        return (Y, Z)

    def spectral_partition(self, red, blue, Y, Z):

        #Randomly select about half of those indices
        r = np.random.rand(len(Y),1)
        Y1 = Y[np.where(r > .5)[0]]
        Y2 = Y[np.where(r <= .5)[0]]

        #select half rows and quarter (approx) columns randomly from A
        A1 = red[Z, :][:,Y1]
        # zero out all rows and cols of A1 where the degree is > 20d
        d = self.a + (self.k - 1)*self.b
        row_degrees = np.sum(A1, axis=1)
        col_degrees = np.sum(A1, axis=0)
        bad_rows = np.where(row_degrees > 20*d)[0]
        bad_cols = np.where(col_degrees > 20*d)[0]
        if len(bad_rows)>0:
            A1[bad_rows,:] = 0

        if len(bad_cols)>0:
            A1[:,bad_cols] = 0
        sparse.csr_matrix.eliminate_zeros(A1)
        #singular value decomposition of A1
        U,S,V_t = svds(A1, k=self.k)
        
        #get k largest singular values
        # max_S = np.argsort(S)[len(S) - self.k:]
        # U = U[:,max_S]
        #Get k equally spaced columns from Y2 - idxs not used in svd
        colsY2 = Y2[np.random.randint(0, len(Y2)-1, size=2*int(np.log2(self.n)), dtype= np.int16)]
        #A2 has same rows as A1, k columns not in A1
        A2 = red[Z,:][:,colsY2].todense() - ((self.a+self.b)/(2*self.n))
        # A2 = sparse.csr_matrix(A2)
        #Projection of A2 onto singular values U of A1
        projY2 = np.dot(np.dot(U,U.T),A2)
        #Construct group_top_coordinates: idx is top n/2k coordinates of each proj vector
        group_top_coordinates = np.zeros((2*int(np.log2(self.n)),int(self.n/(2*self.k))))
        group_blue_densities = []
        for i in range(2*int(np.log2(self.n))):
            e1 = np.array(projY2[:,i]).flatten()
            idx = np.argsort(-e1.T)[:int(self.n/(2*self.k))]
            group_top_coordinates[i,:] = Z[idx]
            # calculate the blue density among each group
            group_blue_densities.append(np.sum(blue[Z[idx],:]))
        idx = np.argsort(-np.array(group_blue_densities))[:len(group_blue_densities)//2]
        # purge half of the set with the lowest blue edge density
        group_top_coordinates = group_top_coordinates[idx,:]

        #Find groups with smallest intersection
        tried_combos = {}
        outCxs = None
        combos = list(itertools.combinations(range(len(group_top_coordinates)), self.k))
        combo_score = np.zeros(len(combos))
        for index, combo in enumerate(combos):
            good_combo = True
            for i in combo:
                for j in combo:
                    if j<=i:
                        continue
                    if (i,j) not in tried_combos:
                        tried_combos[(i,j)]=len(np.intersect1d(group_top_coordinates[i,:],group_top_coordinates[j,:]))
                    if tried_combos[(i,j)]>= .2*self.n/(2*self.k):
                        good_combo = False
                    combo_score[index] += tried_combos[(i,j)]**2
                    #combo_score[index] = max(combo_score[index],tried_combos[(i,j)])
            if good_combo:
                outCxs = group_top_coordinates[combo,:].astype(int)
                break

        if outCxs is None:
            top_combo = np.argmin(combo_score)
            outCxs = group_top_coordinates[combos[top_combo],:].astype(int)
            print("oops")
            # outCxs = []
            # for block1 in range(self.k):
            #   outCx = np.setdiff1d(group_top_coordinates[block1,:],group_top_coordinates[list(range(block1)),:]).astype(int)
            #   outCxs.append(outCx)

        return outCxs, Z, Y

    def plot_output(self, A, A_shuffled, recovered):
        f = plt.figure("original blocked")
        # A = A.todense()
        # A_shuffled = A_shuffled.todense()
        # recovered = recovered.todense()
        markersize = 0.01
        plt.spy(A, markersize=markersize);
        # plt.colorbar()

        f = plt.figure("shuffled input")
        plt.spy(A_shuffled, markersize=markersize);
        # plt.colorbar()
        f = plt.figure("recovered output")
        plt.spy(recovered, markersize=markersize);
        # plt.colorbar()
        plt.show()




    def merge(self, outCxs, A, Y, blue):
        dZZ = (self.a+self.b)/8
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
            xy = []
            yx = []
            for block2 in range(self.k):
                if block2 == block1:
                    continue

                xy = np.union1d(xy, bad_xy[(block1, block2)])
                yx = np.union1d(yx, bad_xy[(block2, block1)])

            outCx = np.setdiff1d(outCxs[block1], xy).astype(int)
            outCx = np.union1d(outCx, yx).astype(int)
            newOutCxs.append(outCx)


        dxs = []

        for outCx in outCxs:
            C1 = blue[outCx,:][:,Y]
            dxs.append(np.sum(C1, axis=0))

        idY = np.argmax(dxs, axis = 0)[0]
        indices = np.array([])
        for i in range(len(outCxs)):
            outCx = np.union1d(outCxs[i],Y[np.where(idY == i)[0]])
            # np.random.shuffle(outCx)
            indices = np.concatenate((indices, outCx))
        indices = indices.astype(int)
        return A[indices,:][:,indices]




    def correction(self, outCxs, A, Z):
        extra = Z
        for outCx in outCxs:
            extra = np.setdiff1d(extra, outCx).astype(int)
        dxs = []
        for outCx in outCxs:
            Cx = A[outCx,:][:,extra]
            dx = np.sum(Cx, axis = 0)
            dxs.append(dx)
        
        idextra = np.argmax(dxs, axis = 0)[0]

        outCxs_list = []
        for i in range(len(outCxs)):
            outCx = np.union1d(outCxs[i], extra[np.where(idextra == i)[0]]).astype(int)
            outCxs_list.append(outCx)
        return outCxs_list




if __name__ == "__main__":
    # beginning_of_time = time.time()
    # num_test = 10
    # for i in range(num_test):
    #     s = SBM(4, 1000, 50, 5)
    #     s.run()
    # print("Running %s takes: %s sec" %(num_test, time.time() - beginning_of_time))
    s = SBM(4, 4000, 50, 5)
    s.run()







