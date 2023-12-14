import cv2
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import NearestCentroid
from skimage.measure import block_reduce
import warnings
warnings.filterwarnings("ignore")

#define nearest centroid classifier
def nearest_centroid_classifier(train_samples, test_samples, lda=None):
    ## get the L1, L2, COS distance between f and fis
    def get_distance_matrix(test_sample, centroids):
        distance_matrix = np.zeros((centroids.shape[0], 3))
        
        #get distance measurements for each centroid
        for i in range(centroids.shape[0]):
            centroid = centroids[i].reshape(centroids.shape[1], 1)
            d1 = abs(test_sample - centroid).sum()
            d2 = np.sum((test_sample - centroid) * (test_sample - centroid))
            d3 = 1 - (test_sample.T.dot(centroid) / (np.linalg.norm(test_sample) * np.linalg.norm(centroid)))
            distance_matrix[i, 0] = d1
            distance_matrix[i, 1] = d2
            distance_matrix[i, 2] = d3

        return distance_matrix
    
    #establish variables for lda
    test_samples_transformed = test_samples
    centroids = block_reduce(train_samples, (3,1), np.mean)  # use mean to calculate the centroids
    
    #parameter to discern between lda or original
    if lda is not None:
        test_samples_transformed = lda.transform(test_samples_transformed)
        centroids = lda.transform(centroids)
    
    #initialize lists for predictions and distances
    predictions = []
    distance_values = []
    
    #loop through the test samples and get predictions
    for i in range(test_samples_transformed.shape[0]):
        test_sample = test_samples_transformed[i, :].reshape(test_samples_transformed.shape[1], 1)
        distance_matrix = get_distance_matrix(test_sample, centroids)
        min_distance_index = np.argmin(distance_matrix, axis=0)
        prediction = min_distance_index + 1

        values = []
        value_matrix = distance_matrix[min_distance_index]
        values.append(value_matrix[0][0])
        values.append(value_matrix[1][1])
        values.append(value_matrix[2][2])

        predictions.append(prediction)
        distance_values.append(values)

    return (distance_values, predictions)


#function to get crr rates
def evaluate(predictions, true_labels):
    correct_classifications = np.zeros(3)
    #loop through targets
    for i in range(len(true_labels)):
        prediction = predictions[i]
        if prediction[0] == true_labels[i]:
            correct_classifications[0] += 1
        if prediction[1] == true_labels[i]:
            correct_classifications[1] += 1
        if prediction[2] == true_labels[i]:
            correct_classifications[2] += 1

    return correct_classifications / len(true_labels)
