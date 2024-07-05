import math

import numpy as np
from PIL import Image
import skimage
from skimage import color, io
from skimage.io import imread
from skimage.color import rgb2gray
from math import sqrt

def load(image_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.

    Args:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None
    
    out = skimage.io.imread(image_path)
    ### YOUR CODE HERE
    # Use skimage io.imread
    pass
    ### END YOUR CODE

    # Let's convert the image to be between the correct range.
    out = out.astype(np.float64) / 255
    return out


    

def dim_image(image):
    """Change the value of every pixel by following

                        x_n = 0.5*x_p^2

    where x_n is the new value and x_p is the original value.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    
    image_height, image_width, pixel_val = image.shape
    out = np.empty([image_height, image_width, pixel_val], dtype=float)
       
    for i in range(image_width):
        for j in range(image_height): 
            for k in range(pixel_val): 
                out[i][j][k] = 0.5 * image[i][j][k]* image[i][j][k]
      
    
    pass

    return out


def convert_to_grey_scale(image):
    """Change image to gray scale.

    HINT: Look at `skimage.color` library to see if there is a function
    there you can use.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width).
    """
    
    out = rgb2gray(image)

    pass
   
    return out


def rgb_exclusion(image, channel):
    """Return image **excluding** the rgb channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "R", "G" or "B".

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    image_height, image_width, pixel_val = image.shape
    out = np.empty([image_height, image_width, pixel_val], dtype=float)

    if channel == 'R':
          for i in range(image_height):
                for j in range(image_width):
                    for k in range(pixel_val):
                         if k == 0:
                                out[i][j][k] = 0.0
                         else: 
                                out[i][j][k] = image[i][j][k]
    elif channel == 'G':
          for i in range(image_height):
                for j in range(image_width):
                    for k in range(pixel_val):
                         if k == 1:
                                out[i][j][k] = 0.0
                         else: 
                                out[i][j][k] = image[i][j][k]
    elif channel == 'B':
         for i in range(image_height):
                for j in range(image_width):
                    for k in range(pixel_val):
                         if k == 2:
                                out[i][j][k] = 0.0
                         else: 
                                out[i][j][k] = image[i][j][k]
    else:
        print("Error: channel does not exist")
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out


def lab_decomposition(image, channel):
    """Decomposes the image into LAB and only returns the channel specified.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "L", "A" or "B".

    Returns:
        out: numpy array of shape(image_height, image_width).
    """

    lab = color.rgb2lab(image)
    out = lab

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out


def hsv_decomposition(image, channel='H'):
    """Decomposes the image into HSV and only returns the channel specified.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "H", "S" or "V".

    Returns:
        out: numpy array of shape(image_height, image_width).
    """
    image_height, image_width, pixel_val = image.shape
    hsv = color.rgb2hsv(image)
    #out = np.empty([image_height, image_width, pixel_val], dtype=float)
    if channel == 'H':
        out = hsv[:, :, 0]
    elif channel == 'S':
        out = hsv[:, :, 1]
    elif channel == 'V':
        out = hsv[:, :, 2]
    else:
        print("Error: channel does not exist")
    
    #out = None

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out


def mix_images(image1, image2, channel1, channel2):
    """Combines image1 and image2 by taking the left half of image1
    and the right half of image2. The final combination also excludes
    channel1 from image1 and channel2 from image2 for each image.

    HINTS: Use `rgb_exclusion()` you implemented earlier as a helper
    function. Also look up `np.concatenate()` to help you combine images.

    Args:
        image1: numpy array of shape(image_height, image_width, 3).
        image2: numpy array of shape(image_height, image_width, 3).
        channel1: str specifying channel used for image1.
        channel2: str specifying channel used for image2.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    image_height, image_width, pixel_val = image1.shape
    out = np.empty([image_height, image_width, pixel_val], dtype=float)
    for i in range(image_height):
        for j in range(image_width):
            for k in range(pixel_val):
                if j in range(int(image_width/2)):
                    out[i][j][k] = image1[i][j][k]
                else: 
                    out[i][j][k] = image2[i][j][k]
   

    if channel1 == 'R': 
        for i in range(image_height):
            for j in range(int(image_width/2)):
                for k in range(pixel_val):
                    if k == 0:
                        out[i][j][k] = 0.0
                    else:
                        out[i][j][k] = image1[i][j][k]
    elif channel1 == 'G': 
        for i in range(image_height):
            for j in range(int(image_width/2)):
                for k in range(pixel_val):
                    if k == 1:
                        out[i][j][k] = 0.0
                    else:
                        out[i][j][k] = image1[i][j][k] 
    elif channel1 == 'B': 
        for i in range(image_height):
            for j in range(int(image_width/2)):
                for k in range(pixel_val):
                    if k == 1:
                        out[i][j][k] = 0.0
                    else:
                        out[i][j][k] = image1[i][j][k] 
    else: 
        print("Error: channel undefined")
        
        
        
    if channel2 == 'R': 
        for i in range(image_height):
            for j in range(int(image_width/2), image_width):
                for k in range(pixel_val):
                    if k == 0:
                        out[i][j][k] = 0.0
                    else:
                        out[i][j][k] = image2[i][j][k]
    elif channel2 == 'G': 
        for i in range(image_height):
            for j in range(int(image_width/2), image_width):
                for k in range(pixel_val):
                    if k == 1:
                        out[i][j][k] = 0.0
                    else:
                        out[i][j][k] = image2[i][j][k] 
    elif channel2 == 'B': 
        for i in range(image_height):
            for j in range(int(image_width/2), image_width):
                for k in range(pixel_val):
                    if k == 1:
                        out[i][j][k] = 0.0
                    else:
                        out[i][j][k] = image1[i][j][k] 
    else: 
        print("Error: channel undefined")
                    
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out


def mix_quadrants(image):
    """THIS IS AN EXTRA CREDIT FUNCTION.

    This function takes an image, and performs a different operation
    to each of the 4 quadrants of the image. Then it combines the 4
    quadrants back together.

    Here are the 4 operations you should perform on the 4 quadrants:
        Top left quadrant: Remove the 'R' channel using `rgb_exclusion()`.
        Top right quadrant: Dim the quadrant using `dim_image()`.
        Bottom left quadrant: Brighthen the quadrant using the function:
            x_n = x_p^0.5
        Bottom right quadrant: Remove the 'R' channel using `rgb_exclusion()`.

    Args:
        image1: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    image_height, image_width, pixel_val = image.shape
    out = np.empty([image_height, image_width, pixel_val], dtype=float)
    #out = image
    for i in range(image_height):
         for j in range(image_width):
            for k in range(pixel_val):
                if i in range(int(image_height/2)) and j in range(int(image_width/2)):   
                    if k == 0:
                        out[i][j][k] = 0.0
                    else:
                        out[i][j][k] = image[i][j][k]
                    
                elif image_height/2 <= i < image_height and 0 <= j < image_width/2: 
                    out[i][j][k] = sqrt(image[i][j][k])
                    
                elif 0 <= i < image_height/2 and image_width/2 <= j < image_width: 
                    out[i][j][k] = 0.5*image[i][j][k]*image[i][j][k]
                else: 
                    if k == 0:
                        out[i][j][k] = 0.0
                    else:
                        out[i][j][k] = image[i][j][k]
          
    

    # out = None

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out
