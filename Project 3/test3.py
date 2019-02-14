import scipy.io as sio
import numpy as np
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.metrics import mean_squared_error, make_scorer
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import accuracy_score, precision_score
from skimage.feature import hog
import cv2
import os

def get_postive_labels(x,y):
    returned_matrix = np.zeros((int(x.shape[0]/2), x.shape[1]))
    index = 0
    for i in range(0, int(np.shape(y)[0]/2)):

        returned_matrix[i,:] = x[index+1] - x[index,:]
        index = index+2
    labels = np.ones(int(np.shape(y)[0]/2))
    return returned_matrix, labels

def get_negative_labels(x,y):
  
    returned_matrix = np.zeros((int(x.shape[0]/2), x.shape[1]))
    index = 0
    for i in range(0, int(np.shape(y)[0]/2)):

        returned_matrix[i,:] = x[index,:] - x[index+1,:]
        index = index+2
    labels = np.ones(int(np.shape(y)[0]/2))*-1
    return returned_matrix, labels

def read_folder_and_compute_hog(folder, length_indicated):
    i = 0
    
    hog_array = np.zeros((length_indicated,30752))
    for filename in os.listdir(folder):
        print(filename)
        image = cv2.imread(folder + filename)

        hog_features = hog(image, orientations=32, pixels_per_cell=(16, 16), multichannel= True, cells_per_block=(1, 1), feature_vector = True)

        hog_array[i,:] = hog_features

        i = i+1

    return hog_array

train_anno_sen = sio.loadmat('stat-sen.mat')
train_anno_gov = sio.loadmat('stat-gov.mat')

features_sen = train_anno_sen['face_landmark']
features_gov = train_anno_gov['face_landmark']

difference_sen = train_anno_sen['vote_diff']
difference_gov = train_anno_gov['vote_diff']


hog_sen = read_folder_and_compute_hog('img-elec/senator/', 116)
hog_gov = read_folder_and_compute_hog('img-elec/governor/', 112)

features_sen = np.concatenate((features_sen, hog_sen), axis = 1)
features_gov = np.concatenate((features_gov, hog_gov), axis = 1)


features_sen_pos = features_sen[0:58,:]
features_sen_neg = features_sen[58:,:]
difference_sen_pos = difference_sen[0:58]
difference_sen_neg = difference_sen[58:]

features_gov_pos = features_gov[0:56, :]
features_gov_neg = features_gov[56:,:]
difference_gov_pos = difference_gov[0:56]
difference_gov_neg = difference_gov[56:]

sen_pos, labels_sen_pos = get_postive_labels(features_sen_pos, difference_sen_pos)
sen_neg, labels_sen_neg = get_negative_labels(features_sen_neg, difference_sen_neg)

gov_pos, labels_gov_pos = get_postive_labels(features_gov_pos, difference_gov_pos)
gov_neg, labels_gov_neg = get_negative_labels(features_gov_neg, difference_gov_neg)


full_sen = np.concatenate((sen_pos,sen_neg), axis = 0)
full_gov = np.concatenate((gov_pos,gov_neg), axis = 0)


labels_sen = np.concatenate((labels_sen_pos, labels_sen_neg))
labels_gov = np.concatenate((labels_gov_pos, labels_gov_neg))



full_sen = minmax_scale(full_sen, axis= 0)
full_gov = minmax_scale(full_gov, axis = 0)


X_train, X_test, y_train, y_test = train_test_split(full_gov, labels_gov, test_size=0.2, shuffle=True, random_state = 20)

#gamma = np.linspace(2**-17,2**5,10).tolist()
C = (2**np.linspace(-5,13, num=10)).tolist()

tuned_parameters = [{'C': C}]

#for i in range(0,1):
clf = GridSearchCV(LinearSVC(fit_intercept= False), tuned_parameters, cv=10,
                    scoring='accuracy', n_jobs = -1)
clf.fit(X_train, y_train)

#train_accuracy.append(accuracy_train)
print(clf.best_score_)
print(clf.best_params_)
test_acc = clf.predict(X_test)
accuracy_test = accuracy_score(y_test, test_acc)
train_acc = clf.predict(X_train)
accuracy_train = accuracy_score(y_train, train_acc)
print(accuracy_test)