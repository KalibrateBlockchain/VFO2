import numpy as np

def whitenMatrix(matrix):
    """Whitening tranformation is applied to the images given as a matrix"""
    """The transformation for the matrix X is given by E*D^(-1/2)*transpose(E)*X"""
    """Where D is a diagonal matrix containing eigen values of covariance matrix of X"""
    """E is the matrix containing eigen vectors of covariance matrix of X"""
    # Covariance matrix is approximated by this
    covMatrix = np.dot(matrix, matrix.T)/matrix.shape[1]

    # Doing the eigen decomposition of cavariance matrix of X 
    eigenValue, eigenVector = np.linalg.eigh(covMatrix)
    # Making a diagonal matrix out of the array eigenValue
    diagMatrix = np.diag(eigenValue)
    # Computing D^(-1/2)
    invSqrRoot = np.sqrt(np.linalg.pinv(diagMatrix))
    # Final matrix which is used for transformation
    whitenTrans = np.dot(eigenVector,np.dot(invSqrRoot, eigenVector.T))
    # whiteMatrix is the matrix we want after all the required transformation
    # To verify, compute the covvariance matrix, it will be approximately identity
    whiteMatrix = np.dot(whitenTrans, matrix)

    # print np.dot(whiteMatrix, whiteMatrix.T)/matrix.shape[1]

    return whiteMatrix

def FOBI(X):
    """Fourth Order Blind Identification technique is used.
    The function returns the unmixing matrix.
    The paper by J. Cardaso is in itself the best resource out there for it.
    SOURCE SEPARATION USING HIGHER ORDER MOMENTS - Jean-Francois Cardoso"""	
    print(X.mean(1).shape)
    X = np.matrix(X.astype('float64'))
    X -= X.mean(1) # centered
    X = whitenMatrix(X) # whitened 
    rows = X.shape[0]
    n = X.shape[1]
    # Initializing the weighted covariance matrix which will hold the fourth order information
    weightedCovMatrix = np.zeros([rows, rows]) 

    # Approximating the expectation by diving with the number of data points
    for signal in X.T:
        norm = np.linalg.norm(signal)
        weightedCovMatrix += norm*norm*np.outer(signal, signal)

    weightedCovMatrix /= n

    # Doing the eigen value decomposition
    eigValue, eigVector = np.linalg.eigh(weightedCovMatrix)

    # print eigVector
    return eigVector

if __name__ == "__main__":
    f1 = '/mnt/cvd/datasets/chile_data_2020_06_30/COVID19-POSITIVE/alphabet-a-z_UPID-0e8aa993_20200702181308.wav'
    f2 = '/mnt/cvd/datasets/chile_data_2020_06_30/COVID19-POSITIVE/alphabet-a-z_UPID-2520714b_20200702182323.wav'
    f3 = '/mnt/cvd/datasets/chile_data_2020_06_30/COVID19-POSITIVE/alphabet-a-z_UPID-2cf12bce_20200702180743.wav'
    from scipy.io import wavfile 
    r1, d1 = wavfile.read(f1)
    r2, d2 = wavfile.read(f2)
    r3, d3 = wavfile.read(f3)
    d1, d2, d3 = d1[:160000].reshape(1,-1), d2[:160000].reshape(1,-1), d3[:160000].reshape(1,-1)
    X = np.concatenate((d1, d2, d3), axis=0)
    #X = np.random.rand(20,20)
    print(X.shape)
    W = FOBI(X)
    print(W.shape)
    Y = W * np.matrix(X)
    X_p = np.linalg.inv(W)*Y
    print(X_p.shape)
#     exit()
    for i in range(X_p.shape[0]):
#         print("****", Y[i].reshape(-1).shape)
        print(X_p[i].astype('int16').reshape(-1,1).shape)
        wavfile.write("example"+str(i)+".wav", r1, X_p[i].astype('int16').reshape(-1,1))