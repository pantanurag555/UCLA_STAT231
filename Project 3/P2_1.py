import os
import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn import svm
from skimage.feature import hog
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

def ld_images(folder):
    images=[]
    for file in os.listdir(folder):
        image = cv2.imread(os.path.join(folder,file))
        if image is not None:
            images.append(image)
    return np.asarray(images)

def feature_diff(data, votes):
    n = data.shape[0]
    ones = np.full((int(n / 4)), 1)
    minus_ones = np.full((int(n / 2) - int(n / 4)), -1)
    labels = np.r_[ones, minus_ones]
    k = 0
    diff = []
    for i in range(0,n,2):
        if(labels[k] == 1):
            diff.append(data[i+1] - data[i])
        else:
            diff.append(data[i] - data[i+1])
        k += 1
    diff = np.array(diff)
    return diff, labels

def plot_graph(train, test, type):
    x = range(1,len(train)+1)
    red_line = mlines.Line2D([], [], color='red', marker='o', label='Train Data')
    green_line = mlines.Line2D([], [], color='green', marker='o', label='Test Data')
    plt.plot(x, train, 'ro-')
    plt.plot(x, test, 'go-')
    plt.title(str(type)+" Graph")
    plt.legend(handles = [red_line, green_line])
    plt.show()

all_gov_rgb = ld_images("img-elec/governor")
all_gov_hog = []
for i in range(len(all_gov_rgb)):
    all_gov_rgb[i] = cv2.cvtColor(all_gov_rgb[i], cv2.COLOR_BGR2RGB)
    fd, hog_image = hog(all_gov_rgb[i], orientations = 32, pixels_per_cell = (16, 16), cells_per_block = (1,1), visualize = True, multichannel = True)
    all_gov_hog.append(fd)

all_sen_rgb = ld_images("img-elec/senator")
all_sen_hog = []
for i in range(len(all_sen_rgb)):
    all_sen_rgb[i] = cv2.cvtColor(all_sen_rgb[i], cv2.COLOR_BGR2RGB)
    fd, hog_image = hog(all_sen_rgb[i], orientations = 32, pixels_per_cell = (16, 16), cells_per_block = (1,1), visualize = True, multichannel = True)
    all_sen_hog.append(fd)

stat_gov = sio.loadmat('stat-gov.mat')
gov_landmark = stat_gov['face_landmark']
gov_vote_diff = stat_gov['vote_diff']

stat_sen = sio.loadmat('stat-sen.mat')
sen_landmark = stat_sen['face_landmark']
sen_vote_diff = stat_sen['vote_diff']

gov_features = np.c_[all_gov_hog, gov_landmark]
sen_features = np.c_[all_sen_hog, sen_landmark]

gov_diff, gov_labels = feature_diff(gov_features, gov_vote_diff)
sen_diff, sen_labels = feature_diff(sen_features, sen_vote_diff)

gov_diff = minmax_scale(gov_diff, axis = 0)
sen_diff = minmax_scale(sen_diff, axis = 0)

gov_train_diff, gov_test_diff, gov_train_labels, gov_test_labels = train_test_split(gov_diff, gov_labels, test_size=0.2, random_state=34)
sen_train_diff, sen_test_diff, sen_train_labels, sen_test_labels = train_test_split(sen_diff, sen_labels, test_size=0.2, random_state=2)

c_range = 2**np.linspace(-5,13, num=10)
train_accuracy = []
test_accuracy = []
best_parameters = []

svc = svm.LinearSVC(fit_intercept = False, loss = 'hinge')
parameters = {'C':c_range}
clf = GridSearchCV(svc, parameters, cv = 10, n_jobs = -1, scoring = 'accuracy', iid=True, verbose = True)
clf.fit(gov_train_diff, gov_train_labels)
train_pred = clf.predict(gov_train_diff)
test_pred = clf.predict(gov_test_diff)
train_accuracy.append(accuracy_score(gov_train_labels, train_pred))
test_accuracy.append(accuracy_score(gov_test_labels, test_pred))
best_parameters.append(clf.best_params_)

svc = svm.LinearSVC(fit_intercept = False, loss = 'hinge')
parameters = {'C':c_range}
clf = GridSearchCV(svc, parameters, cv = 10, n_jobs = -1, scoring = 'accuracy', iid=True, verbose = True)
clf.fit(sen_train_diff, sen_train_labels)
train_pred = clf.predict(sen_train_diff)
test_pred = clf.predict(sen_test_diff)
train_accuracy.append(accuracy_score(sen_train_labels, train_pred))
test_accuracy.append(accuracy_score(sen_test_labels, test_pred))
best_parameters.append(clf.best_params_)

print("Train Accuracy:\n",train_accuracy)
print("Test Accuracy:\n",test_accuracy)
print("Best Parameters:\n",best_parameters)

plot_graph(train_accuracy,test_accuracy, "Accuracy")