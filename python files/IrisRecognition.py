#import packages
import os
import glob 
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IrisLocalization import *
from IrisNormalization import *
from IrisEnhancement import *
from IrisFeatureExtraction import *
from IrisMatching import *
from IrisPerformanceEvaluation import *


#initialize lists to transfer database into pandas dataframe
train_db = []
test_db = []
train_location = []
test_location = []

# Construct the database directory path
database_dir = os.path.join(os.getcwd(), 'CASIA Iris Image Database (version 1.0)')

#populate lists for dataframe
for i in range(1, 109, 1):
    person_index = '{:03}'.format(i)
    person_imgs = os.path.join(database_dir, person_index)
    train_dir = os.path.join(person_imgs, "1")
    test_dir = os.path.join(person_imgs, "2")

    train_list = glob.glob(os.path.join(train_dir, "*.bmp"))
    test_list = glob.glob(os.path.join(test_dir, "*.bmp"))

    train_location += train_list
    test_location += test_list

    for img_train in train_list:
        img_flat = cv2.imread(img_train, 0).ravel()
        train_db.append(img_flat)

    for img_test in test_list:
        img_flat = cv2.imread(img_test, 0).ravel()
        test_db.append(img_flat)
        
#create train and test dataframes
train = pd.DataFrame(train_db, columns=[i for i in range(280*320)]) # the image dimension is 280*320
test = pd.DataFrame(test_db, columns=[i for i in range(280*320)]) # the image dimension is 280*320

#create list of training images from the dataframe
train_lst = []
for i in range(train.shape[0]):
    im = train.iloc[i]
    image_data = im.values
    image = image_data.reshape((280, 320))
    train_lst.append(image)
    
#create list of testing images from the dataframe
test_lst = []
for i in range(test.shape[0]):
    im = test.iloc[i]
    image_data = im.values
    image = image_data.reshape((280, 320))
    test_lst.append(image)
    
#get train and test localized iris images
train_iris_lst, train_pupil_c_lst = execute_iris_region(train_lst)
test_iris_lst, test_pupil_c_lst = execute_iris_region(test_lst)

#get normalized train and test iris images
train_normalization_lst = execute_normalization(train_iris_lst, train_pupil_c_lst)
test_normalization_lst = execute_normalization(test_iris_lst, test_pupil_c_lst)

#get enhanced train and test iris images
train_ROI_lst = execute_enhancement(train_normalization_lst)
test_ROI_lst = execute_enhancement(test_normalization_lst)

#get train and test feature vectors
training_fc_lst = execute_feature_vector(train_ROI_lst)
testing_fc_lst = execute_feature_vector(test_ROI_lst)

#prepare train and test data
#training_fc_lst is a list with 324 1d list entries each 1d list entry has length of 1536
#testing_fc_lst is a list with 432 1d list entries each 1d list entry has length of 1536
train_X = np.array(training_fc_lst)
train_Y = np.array([(i//3+1) for i in range(train_X.shape[0])])
test_X = np.array(testing_fc_lst)
test_Y = np.array([(i//4+1) for i in range(test_X.shape[0])])

#original centroid prediction
values_Org, predictions_Org = nearest_centroid_classifier(train_X, test_X, lda=None)
#get LDA predictions
lda_org = LDA().fit(train_X, train_Y)   
values_LDA, predictions_LDA = nearest_centroid_classifier(train_X, test_X, lda=lda_org)

#get variables ready for table
originals = evaluate(predictions_Org, test_Y)
transforms1 = evaluate(predictions_LDA, test_Y)
plot_CRR(originals, transforms1)

#plot ROC curve
n_arr = [i for i in range(10, 108, 10)]+[107]
recognition_rates = np.empty(len(n_arr))
for i in range(len(n_arr)):
    lda = LDA(n_components=n_arr[i])
    lda.fit(train_X, train_Y) 
    max_recognition_rate = np.amax(evaluate(nearest_centroid_classifier(train_X, test_X, lda=lda)[1], test_Y))
    recognition_rates[i] = max_recognition_rate

plot_LDA_tunning(n_arr, recognition_rates)
