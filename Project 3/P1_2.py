import os
import cv2
import pickle
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn import svm
from skimage.feature import hog
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import GridSearchCV

def ld_images(folder):
    images=[]
    for file in os.listdir(folder):
        image = cv2.imread(os.path.join(folder,file))
        if image is not None:
            images.append(image)
    return np.asarray(images)

def plot_graph(train, test, type):
    x = range(1,len(train)+1)
    red_line = mlines.Line2D([], [], color='red', marker='o', label='Train Data')
    green_line = mlines.Line2D([], [], color='green', marker='o', label='Test Data')
    plt.plot(x, train, 'ro-')
    plt.plot(x, test, 'go-')
    plt.title(str(type)+" Graph")
    plt.legend(handles = [red_line, green_line])
    plt.show()

all_images_rgb = ld_images("img")
all_images_hog = []
for i in range(len(all_images_rgb)):
    all_images_rgb[i] = cv2.cvtColor(all_images_rgb[i], cv2.COLOR_BGR2RGB)
    fd, hog_image = hog(all_images_rgb[i], orientations = 32, pixels_per_cell = (16, 16), cells_per_block = (1,1), visualize = True, multichannel = True)
    all_images_hog.append(fd)

train_anno = sio.loadmat('train-anno.mat')
face_landmark = train_anno['face_landmark']
trait_annotation = train_anno['trait_annotation']

total_features = np.c_[all_images_hog, face_landmark]

total_features = minmax_scale(total_features, axis = 0)

thresholds = np.mean(trait_annotation, axis = 0)
trait_labels = np.array([[1 if x >= 0 else -1 for x in trait_annotation[:,i]] for i in range(trait_annotation.shape[1])])
trait_labels = trait_labels.T

division = int(0.8 * trait_labels.shape[0])
train_data = total_features[:division,]
train_reg = trait_annotation[:division,]
train_labels = trait_labels[:division,]
test_data = total_features[division:,]
test_reg = trait_annotation[division:,]
test_labels = trait_labels[division:,]

c_range = 2**np.linspace(-5,13,10)
p_range = 2**np.linspace(-9,1,6)
gamma_range = 2**np.linspace(-17,5,12)
mean_sq_err = []
train_accuracy = []
train_precision =[]
test_accuracy = []
test_precision =[]
best_parameters = []
best_models = []
for i in range(14):
    svr = svm.SVR(kernel='rbf')
    parameters = {'C':c_range, 'epsilon':p_range, 'gamma':gamma_range}
    clf = GridSearchCV(svr, parameters, cv = 5, scoring = 'neg_mean_squared_error', n_jobs = 6, iid=True, verbose = True)
    clf.fit(train_data,train_reg[:,i])
    best_models.append(clf)
    mean_sq_err.append(-1 * clf.best_score_)
    train_pred = clf.predict(train_data)
    train_pred_labels = np.array([1 if x >= thresholds[i] else -1 for x in train_pred])
    test_pred = clf.predict(test_data)
    test_pred_labels = np.array([1 if x >= thresholds[i] else -1 for x in test_pred])
    train_accuracy.append(np.sum((train_pred_labels * train_labels[:,i]) > 0) / train_pred_labels.shape[0])
    test_accuracy.append(np.sum((test_pred_labels * test_labels[:,i]) > 0) / test_pred_labels.shape[0])
    train_precision.append(np.sum((train_pred_labels * (train_labels[:,i] > 0)) > 0) / np.sum(train_pred_labels > 0))
    test_precision.append(np.sum((test_pred_labels * (test_labels[:,i] > 0)) > 0) / np.sum(test_pred_labels > 0))
    best_parameters.append(clf.best_params_)
print("Train Accuracy:\n",train_accuracy)
print("Test Accuracy:\n",test_accuracy)
print("Train Precision:\n",train_precision)
print("Test Precision:\n",test_precision)
print("Mean Squared Error:\n",mean_sq_err)
print("Best Parameters:\n",best_parameters)

best_models = np.array(best_models)
pickle.dump(best_models, open('1.2_best_models.pkl', 'wb'))
plot_graph(train_accuracy,test_accuracy, "Accuracy")
plot_graph(train_precision,test_precision, "Precision")