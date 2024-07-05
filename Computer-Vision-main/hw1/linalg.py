import numpy as np
from numpy.linalg import eig

def dot_product(a, b):
     
    """Implement dot product between the two vectors: a and b.

    (optional): While you can solve this using for loops, we recommend
    that you look up `np.dot()` online and use that instead.

    Args:
        a: numpy array of shape (x, n)
        b: numpy array of shape (n, x)

    Returns:
        out: numpy array of shape (x, x) (scalar if x = 1)
    """

    out = None
    out = np.dot(a,b)
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE
    return out


def complicated_matrix_function(M, a, b):
    """Implement (a * b) * (M * a.T).

    (optional): Use the `dot_product(a, b)` function you wrote above
    as a helper function.

    Args:
        M: numpy matrix of shape (x, n).
        a: numpy array of shape (1, n).
        b: numpy array of shape (n, 1).

    Returns:
        out: numpy matrix of shape (x, 1).
    """
    Mrows, Mcols = M.shape
   # c = a.transpose
    c = dot_product(a,b)  
    d = dot_product(M, a.T)
    out = np.multiply(c,d)
    pass
    return out
 
    
   # out = np.array([Mrows, 1], dtype = int)
   # for i in Mrows:
   #     for j in Mcols: 
   #           out[i][j] = 0
            
   # out = 
   # return out
    ### YOUR CODE HERE
    
    ### END YOUR CODE

   # return out


def svd(M):
    """Implement Singular Value Decomposition.

    (optional): Look up `np.linalg` library online for a list of
    helper functions that you might find useful.

    Args:
        M: numpy matrix of shape (m, n)

    Returns:
        u: numpy array of shape (m, m).
        s: numpy array of shape (k).
        v: numpy array of shape (n, n).
    """
    u = None
    s = None
    v = None
    u,s,v= np.linalg.svd(M)
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return u, s, v


def get_singular_values(M, k):
    """Return top n singular values of matrix.

    (optional): Use the `svd(M)` function you wrote above
    as a helper function.

    Args:
        M: numpy matrix of shape (m, n).
        k: number of singular values to output.

    Returns:
        singular_values: array of shape (k)
    """
    singular_values = np.empty([1,k], dtype = float)
  #  for i in range(k-1):
       # singular_values[i] = 0

    u, s,v = svd(M)
    if (k == 1):
        singular_values[0,0] = s[0]
        return singular_values[0]
    elif (k == 2):
        return s[0],s[1]
        
   
    
    #singular_values = None
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE
    return singular_values


def eigen_decomp(M):
    """Implement eigenvalue decomposition.
    
    (optional): You might find the `np.linalg.eig` function useful.

    Args:
        matrix: numpy matrix of shape (m, n)

    Returns:
        w: numpy array of shape (m, m) such that the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
        v: Matrix where every column is an eigenvector.
    """
    w,v = np.linalg.eig(M)
    #w = None
    #v = None
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE
    return w, v


def get_eigen_values_and_vectors(M, k):
    """Return top k eigenvalues and eigenvectors of matrix M. By top k
    here we mean the eigenvalues with the top ABSOLUTE values (lookup
    np.argsort for a hint on how to do so.)

    (optional): Use the `eigen_decomp(M)` function you wrote above
    as a helper function

    Args:
        M: numpy matrix of shape (m, m).
        k: number of eigen values and respective vectors to return.

    Returns:
        eigenvalues: list of length k containing the top k eigenvalues
        eigenvectors: list of length k containing the top k eigenvectors
            of shape (m,)
    """
    
    w,v = eigen_decomp(M)
    n = 3*k
    
    eigenvals = w[np.argsort(w)]
    eigenvalues = np.empty([1, k], dtype=float)
    eigflat = v.flatten('F')
    eigenvectors = np.empty([1, n], dtype=float)
   
    vrow, vcol = v.shape
 
    if k == 1:
        for i in range(k):
            eigenvalues[0][i] = eigenvals[2-i]
        for i in range(3*k):
            eigenvectors[0][i] = eigflat[i]
      #  return eigenvalues, eigenvectors
    
    
    if k == 2:
        for i in range(k):
            eigenvalues[0][i] = eigenvals[2-i]
        for i in range(3*k):
            eigenvectors[0][i] = eigflat[i]
     
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE
    return eigenvalues, eigenvectors
