# IrisRecognition
Implementation of Li Ma Iris Recognition algorithm

This repository contains 7 python files that outline the algorithm indicated in the LiMa2003.pdf paper. It also contains a notebook which is all the python files used together with some visualization throughout the process.

# Design Explanation:

We start be splitting the training and testing data and storing the image values into pandas dataframes where each row is an image and each column is a value of individual pixels. From there we iterate through the dataframes and create a training and testing list that contains all of the training and testing images. 

We then create a process for iris localization and iterate through the training and testing lists creating a new training and testing list which contain all of the localized iris images.

We then create a process for iris normalization and iterate through the training and testing iris localization lists creating new training and testing lists which contain all of the normalized iris images.

We then create a process for iris enhancement and iterate through the training and testing normalized lists creating new training and testing lists which contain all of the enhanced iris images.

Then we apply our feature extraction process to the two lists resulting in a training and testing feature vector list.

From here we apply a nearest centroid classifier which calculates the distance between two matrices. We use this to find the distances between the predicted values from our LDA model and the test values in the three different distance metrics provided in the paper.

We then create a function to evaluate the accuracy of our LDA and visualize our evaluations.

# Limitations 

It seems that there is work to be done in feature extraction as well as iris enhancement. There could be more tranformations applied in the iris enhancement that could allow for the feature vectors to be more well defined. There could also be modifications to the kernel application of the filters applied to get the feature vectors. Perhaps looking at application depth could be beneficial.
