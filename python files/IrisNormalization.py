import cv2
import numpy as np

#function to convert image coordinates to polar for normalization
def cartesian_to_polar(img, center):
    max_radius = int(np.linalg.norm(img.shape - np.array(center)))
    
    # Create an output image with the same width as the input image and height equal to the maximum radius
    polar_img = np.zeros((max_radius, img.shape[1], 3), dtype=np.uint8)
    
    for i in range(img.shape[1]):  # For each column in the input image
        for j in range(max_radius):  # For each row in the polar image
            theta = 2 * np.pi * i / img.shape[1]  # Calculate the angle
            r = j  # Radius is just the row index in the polar image
            x = center[0] + r * np.cos(theta)
            y = center[1] + r * np.sin(theta)
            
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                polar_img[j, i] = img[int(y), int(x)]
                
    return polar_img

#crop the normalized image
def crop_black_region(image):
    # Find the rows where all values are not black
    rows = np.any(image != 0, axis=1)
    cols = np.any(image != 0, axis=0)
    
    # Find the bounds of the non-black region
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # Slice and return the cropped region
    cropped_image = image[rmin:rmax+1, cmin:cmax+1]
    final_image = cv2.resize(cropped_image, (512,64))
    return final_image

#function to run all normalization functions
def execute_normalization(img_lst, pupil_lst):
    normalization_lst = []
    #normalize every image in the list
    for i in range(len(img_lst)):
        iris_region = img_lst[i]
        pupil_c = pupil_lst[i]
        polar_pic = cartesian_to_polar(iris_region, pupil_c)
        cropped_polar = crop_black_region(polar_pic)
        normalization_lst.append(cropped_polar)
    return normalization_lst
