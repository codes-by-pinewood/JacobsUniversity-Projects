import numpy as np
import math
from collections import deque

def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args -
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns -
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')
    
    # flip the kernel horizontally and vertically 
    kernel = np.flip(kernel, 1)
    kernel = np.flip(kernel, 0)
    
    for m in range(Hi):
        for n in range(Wi):
            out[m, n] =  np.sum(padded[m: m+Hk, n: n+Wk] * kernel)

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out

def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.

    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp.

    Args:
        size: int of the size of output matrix.
        sigma: float of sigma to calculate kernel.

    Returns:
        kernel: numpy array of shape (size, size).
    """
    x= sigma

    kernel = np.zeros((size, size))
    k = int((size-1)/2)
    conster = 1/(2*math.pi*x**2)
    
    for i in range(size):
        for j in range(size):
            kernel[i][j] = conster*math.exp((-(i-k)**2-(j-k)**2)/(2*x**2))

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return kernel

def partial_x(image):
    """ Computes partial x-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        image: numpy array of shape (H, W).
    Returns:
        out: x-derivative image.
    """
    D = np.array([[0.5, 0, -0.5]])

    out = None
    #Hi,Wi = image.shape
    #out = np.zeros(Hi,Wi)
    
    out = conv(image,D)

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out

def partial_y(image):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        image: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """
   

    ### YOUR CODE HERE
    D = np.array([[0.5],[0], [-0.5]])
                  

    out = None
    out = conv(image,D)
    ### END YOUR CODE

    return out

def gradient(image):
    """ Returns gradient magnitude and direction of input img.

    Args:
        image: Grayscale image. Numpy array of shape (H, W).

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W).

    Hints:
        - Use np.sqrt and np.arctan2 to calculate square root and arctan
    """
    Hg, Wg = image.shape
    G = np.zeros(image.shape)
    theta = np.zeros(image.shape)
    
    Gx = partial_x(image)
    Gy = partial_y(image)
    
    for i in range(Hg):
        for j in range(Wg): 
            G[i][j] = np.sqrt(pow(Gx[i][j],2) + pow(Gy[i][j],2))
            
    for i in range(Hg):
        for j in range(Wg): 
            theta[i][j] = (np.arctan2(Gy[i][j],Gx[i][j]) * 180/math.pi +180)%360
            
    
    

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return G, theta


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression.

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).

    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).

    Returns:
        out: non-maxima suppressed image.
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45
    
    theta = theta / 180 * np.pi
    dx = ((np.cos(theta) + 0.5) // 1).astype(int)
    dy = ((np.sin(theta) + 0.5) // 1).astype(int)
    padded = np.pad(G, ((1, 1), (1, 1)), mode='constant')
    i = np.indices((H, W)) + 1
    query = (G >= padded[i[0] + dy, i[1] + dx]) & (G >= padded[i[0] - dy, i[1] - dx])
    out[query] = G[query]
        

    ### BEGIN YOUR CODE
    
    #pass
    ### END YOUR CODE

    return out

def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
        high: high threshold(float) for strong edges.
        low: low threshold(float) for weak edges.

    Returns:
        strong_edges: Boolean array which represents strong edges.
            Strong edges are the pixels with the values greater than
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values smaller or equal to the
            higher threshold and greater than the lower threshold.
    """
    Hi,Wi = img.shape
    strong_edges = np.zeros(img.shape, dtype=np.bool)
    weak_edges = np.zeros(img.shape, dtype=np.bool)
    for i in range(Hi):
        for j in range(Wi):
            if (img[i][j] > high):
                strong_edges[i][j] = True
            elif (low < img[i][j] < high):
                weak_edges[i][j] = True
            else : 
                strong_edges[i][j] = False
                weak_edges[i][j] = False
            
    

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x).

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel.
        H, W: size of the image.
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)].
    """
    neighbors = []

    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors

def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W).
        weak_edges: binary image of shape (H, W).
    
    Returns:
        edges: numpy boolean array of shape(H, W).
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W), dtype=np.bool)

    # Make new instances of arguments to leave the original
    # references intact
    weak_edges = np.copy(weak_edges)
    edges = np.copy(strong_edges)
    
    for y in range(H):
        for x in range(W):
            if strong_edges[y, x]:
                q = deque([(y, x)])
                while len(q) > 0:
                    i, j = q.pop()
                    for i1, j1 in get_neighbors(i, j, H, W):
                        if weak_edges[i1, j1] and not edges[i1, j1]:
                            edges[i1, j1] = True
                            q.appendleft((i1, j1))
                        
                        
                        
    


    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return edges

def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W).
        kernel_size: int of size for kernel matrix.
        sigma: float for calculating kernel.
        high: high threshold for strong edges.
        low: low threashold for weak edges.
    Returns:
        edge: numpy array of shape(H, W).
    """
    kernel = gaussian_kernel(kernel_size, sigma)
    smoothed = conv(img, kernel)
    G, theta = gradient(smoothed)
    
    out = non_maximum_suppression(G,theta)
    strong_edges, weak_edges = double_thresholding(out, high, low)
    edge = link_edges(strong_edges, weak_edges)
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return edge


def hough_transform(img):
    """ Transform points in the input image into Hough space.

    Use the parameterization:
        rho = x * cos(theta) + y * sin(theta)
    to transform a point (x,y) to a sine-like function in Hough space.

    Args:
        img: binary image of shape (H, W).
        
    Returns:
        accumulator: numpy array of shape (m, n).
        rhos: numpy array of shape (m, ).
        thetas: numpy array of shape (n, ).
    """
    # Set rho and theta ranges
    W, H = img.shape
    diag_len = int(np.ceil(np.sqrt(W * W + H * H)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0 + 1)
    thetas = np.deg2rad(np.arange(-90.0, 90.0))

    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Initialize accumulator in the Hough space
    accumulator = np.zeros((2 * diag_len + 1, num_thetas), dtype=np.uint64)
    ys, xs = np.nonzero(img)

    # Transform each point (x, y) in image
    # Find rho corresponding to values in thetas
    # and increment the accumulator in the corresponding coordiate.
    ### YOUR CODE HERE
    r = xs.reshape(-1, 1) * cos_t.reshape(1, -1) + ys.reshape(-1, 1) * sin_t.reshape(1, -1)
    r = (r.reshape(-1) + diag_len).astype(int)
    np.add.at(accumulator, (r, np.tile(np.arange(len(thetas)), len(xs))), 1)
    ### END YOUR CODE

    return accumulator, rhos, thetas
