import cv2
import numpy as np
from scipy import signal

#takes in what channel and changes sigma values accordingly
def construct_kernel(channel):
    ksize = 3  # Size of the kernel (should be odd)
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    
    if channel == 1:
        sigma_x = 3
        sigma_y = 1.5
    if channel == 2:
        sigma_x = 4.5
        sigma_y = 1.5
    
    #loop through each kernel position and apply gabor filter
    for x in range(ksize):
        for y in range(ksize):
            x = x - ksize // 2
            y = y - ksize // 2
            #f should be 1 / dy
            f = 1 / sigma_y
            
            #break down gabor filter equation
            gauss_coeff = (1 / (2 * np.pi * sigma_x * sigma_y))
            gauss = np.exp(-.5 * ((x**2/ sigma_x**2) + (y**2/ sigma_y**2)))
            m = np.cos(2 * np.pi * f * np.sqrt(x**2 + y**2))
            value = gauss_coeff * gauss * m
            #assign each place in kernel to new filtered value
            kernel[x, y] = value

    # Normalize the kernel
    kernel /= np.sum(np.abs(kernel))
    return kernel


def get_filtered_ims(im):
    #set kernels for each channel
    kernel1 = construct_kernel(1)
    kernel2 = construct_kernel(2)
    #apply each channel filter onto ROI
    fi1 = signal.convolve2d(im, kernel1, mode='same', boundary='wrap')
    fi2 = signal.convolve2d(im, kernel2, mode='same', boundary='wrap')
    #save in filtered images in list
    flist = [fi1, fi2]
    return flist


def get_feature_vector(im_list):
    feature_vec = []
    #set block size for feature extraction
    block_size = 8
    #get feature vector values for each filtered image
    for im in im_list:
        width = im.shape[1]
        height = im.shape[0]
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                block = im[i:i + block_size, j:j + block_size]
                m = np.mean(np.abs(block))
                sigma = np.std(block)
                feature_vec.append(m)
                feature_vec.append(sigma)
    return feature_vec

##function to run all feature vecotr functions
def execute_feature_vector(enhanced_lst):
    fc_lst = []
    #get feature vector for all images in list
    for image in enhanced_lst:
        filtered_lst = get_filtered_ims(image)
        feature_vector = get_feature_vector(filtered_lst)
        fc_lst.append(feature_vector)
    return fc_lst
