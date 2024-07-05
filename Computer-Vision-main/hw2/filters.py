import numpy as np
from PIL import Image
  


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    
    out = np.zeros((Hi, Wi))

    for i in range(Hi):
        for j in range(Wi):
            for k in range(Hk):
                for l in range(Wk):
                    out[i-1][j-1] += image[i-k][j-l] * kernel[k][l]

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
   
    Hi, Wi = image.shape
    new_height = Hi + 2 *pad_height 
    new_width = Wi + 2 * pad_width

    out = np.zeros((new_height, new_width), dtype='float') 
    
    out[pad_height:pad_height+Hi, pad_width:Wi+pad_width] = image



    ### YOUR CODE HERE
    pass
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    image = zero_pad(image, Hk//2, Wk//2)
    kernel = np.flip(kernel, 0)
    kernel = np.flip(kernel, 1)
    for m in range(Hi):
        for n in range(Wi):
            out[m, n] =  np.sum(image[m: m+Hk, n: n+Wk] * kernel)
    ### END YOUR CODE

    return out
     


def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).//img_gray
        g: numpy array of shape (Hg, Wg).//temp_gray

    Returns:
        out: numpy array of shape (Hf, Wf).
    """
    g = np.flip(g, 0)
    g = np.flip(g, 1)
    out = conv_fast(f, g)
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    g = g - np.mean(g)
    out = cross_correlation(f, g)
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    h = (f - np.mean(f))/np.var(f)
    i = (g - np.mean(g))/np.var(g)
    out = cross_correlation(h, i)
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out
