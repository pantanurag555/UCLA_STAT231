import os
import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn import svm
from skimage.feature import hog
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import GridSearchCV

stat_gov = sio.loadmat('stat-gov.mat')
vote_diff = stat_gov['vote_diff']
print(vote_diff[0:4])