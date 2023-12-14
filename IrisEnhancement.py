import cv2
import numpy as np

#function for iris enhancement
def enhance_iris(image):
    # Ensure the image is grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 1: Estimate Background Illumination
    block_size = 16
    background = cv2.resize(cv2.resize(image, (image.shape[1] // block_size, image.shape[0] // block_size)),
                            (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
    
    # Step 2: Lighting Correction
    corrected_image_float = np.float32(image) - np.float32(background)  # Use float for subtraction
    corrected_image = np.clip(corrected_image_float, 0, 255).astype('uint8')

    # Step 3: Local Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32, 32))
    enhanced_image = clahe.apply(corrected_image)
    ROI = enhanced_image[16:64, 0:512]
    return ROI

#function to run iris enhancement
def execute_enhancement(im_lst):
    enhanced_lst = []
    #enhance each image in the list
    for im in im_lst:
        enhanced = enhance_iris(im)
        enhanced_lst.append(enhanced)
    return enhanced_lst