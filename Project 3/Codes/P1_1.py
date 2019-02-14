import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn import svm
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

def plot_graph(train, test, type):
    x = range(1,len(train)+1)
    red_line = mlines.Line2D([], [], color='red', marker='o', label='Train Data')
    green_line = mlines.Line2D([], [], color='green', marker='o', label='Test Data')
    plt.plot(x, train, 'ro-')
    plt.plot(x, test, 'go-')
    plt.title(str(type)+" Graph")
    plt.legend(handles = [red_line, green_line])
    plt.show()

train_anno = sio.loadmat('train-anno.mat')
face_landmark = train_anno['face_landmark']
trait_annotation = train_anno['trait_annotation']

face_landmark = minmax_scale(face_landmark, axis = 0)

thresholds = np.mean(trait_annotation, axis = 0)

train_data, test_data, train_reg, test_reg = train_test_split(face_landmark, trait_annotation, test_size=0.3, random_state=20)

train_labels = np.array([[1 if x >= thresholds[i] else -1 for x in train_reg[:,i]] for i in range(train_reg.shape[1])])
train_labels = train_labels.T
test_labels = np.array([[1 if x >= thresholds[i] else -1 for x in test_reg[:,i]] for i in range(test_reg.shape[1])])
test_labels = test_labels.T

c_range = 2**np.linspace(-5,13,10)
p_range = 2**np.linspace(-9,1,6)
gamma_range = 2**np.linspace(-17,5,12)
mean_sq_err = []
train_accuracy = []
train_precision =[]
test_accuracy = []
test_precision =[]
best_parameters = []
for i in range(14):
    svr = svm.SVR(kernel='rbf')
    parameters = {'C':c_range, 'epsilon':p_range, 'gamma':gamma_range}
    clf = GridSearchCV(svr, parameters, cv = 10, scoring = 'neg_mean_squared_error', n_jobs = -1, iid=True, verbose=True)
    clf.fit(train_data,train_reg[:,i])
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

plot_graph(train_accuracy,test_accuracy, "Accuracy")
plot_graph(train_precision,test_precision, "Precision")