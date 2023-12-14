#import packages
import os
import glob 
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_contours(img):
        image = img
        #threshold image to find contours
        ret, thresh = cv2.threshold(image, 65, 255, cv2.THRESH_BINARY)
        #find all contours
        contours,_ = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        image_copy = image.copy()
        cx = cy = 0
        xmin = xmax = ymin = ymax = 0
        #find our desired pupil contour
        for cnt in contours:
            if (cv2.contourArea(cnt) < 40000 and cv2.contourArea(cnt) > 700):
                cv2.drawContours(image=image_copy, contours=cnt, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    xmin, ymin, xmax, ymax = cnt[0][0][0], cnt[0][0][1], cnt[0][0][0], cnt[0][0][1]
                    for point in cnt:
                        x, y = point[0]
                        xmin, ymin, xmax, ymax = min(xmin, x), min(ymin, y), max(xmax, x), max(ymax, y)

        #estimate pupil radius by takeing the average of min and max radius from center of contour
        pupil_r = (cy-ymin + cx-xmin + xmax-cx + ymax-cy)/4
        return cx, cy, pupil_r
    
def canny_circles(cx, cy, pupil_r, image):
    #set canny variables
    size=120
    pupil_c = cx, cy
    height, width = image.shape
    x_start = max(cx - size, 0)
    x_end = min(cx + size, width)
    y_start = max(cy - size, 0)
    y_end = min(cy + size, height)
    #crop images for easier canny
    image_crop2 = image[y_start:y_end, x_start:x_end]
    image_crop3 = image_crop2.copy()
    #set canny variables
    cimg = cv2.cvtColor(image_crop2,cv2.COLOR_GRAY2BGR)
    k = 15
    blur = cv2.GaussianBlur(cimg,(k,k),0)
    canny = cv2.Canny(blur,15,5)
    canny[0:60]=0
    canny[50:190,50:190]=0
    #get hough circles
    circles = cv2.HoughCircles(
        canny,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=1000,
        param1=100,
        param2=12,
        minRadius=40,   # Rough estimate, may need to adjust
        maxRadius=125   # Rough estimate, may need to adjust
    )

    if circles is not None:
        # If circles are found, take the first one (you may need to change this logic)
        iris_r = circles[0, :][0][2]
        iris_x = circles[0, :][0][0]
        iris_y = circles[0, :][0][1]
        iris_c = (iris_x, iris_y)
        return pupil_c, iris_c, iris_r, image_crop2.shape
    else:
        # Handle the case where no circles are found
        print("No circles found.")
        # You can return None or some default values, or raise an exception
        return None, None, None, image_crop2.shape
    
#function to make circular masks isolating iris
def create_circular_mask(center, radius, shape):
    Y, X = np.ogrid[:shape[0], :shape[1]]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return mask

#function to apply masks and get iris region isolated
def get_iris_region(pupil_c, pupil_r, iris_r, image):
    pupil_mask = create_circular_mask(pupil_c, pupil_r, image.shape)
    iris_mask = create_circular_mask(pupil_c, iris_r, image.shape)

    # Subtract masks to get the iris region
    iris_only_mask = iris_mask.astype(np.uint8) - pupil_mask.astype(np.uint8)

    # Use the iris only mask to extract the iris region from the image
    iris_region = cv2.bitwise_and(image, image, mask=iris_only_mask.astype(np.uint8))
    return iris_region



#function to run all localization functions
def execute_iris_region(imlist):
    iris_list = []
    pupil_center_list = []
    #loop through and localize every image in list
    for image in imlist:
        cx, cy, pupil_r = get_contours(image)
        results = canny_circles(cx, cy, pupil_r, image)
        
        # Check if canny_circles returned None values
        if results[0] is None or results[1] is None or results[2] is None:
            continue  # Skip the current image and continue with the next one

        pupil_c, iris_c, iris_r, shape = results
        #append pupil center to list for normalization part
        pupil_center_list.append(pupil_c)
        iris = get_iris_region(pupil_c, pupil_r, iris_r, image)
        iris_list.append(iris)
    return iris_list, pupil_center_list
