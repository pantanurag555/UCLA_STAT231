import numpy as np
import cv2
import skimage
import os
import matplotlib.pyplot as plt
import scipy.io as sio
import mywarper
import matplotlib.colors as colors
from sklearn.preprocessing import StandardScaler
from skimage import color
from sklearn.model_selection import train_test_split

def ld_landmarks(folder):
    matrices=[]
    for file in os.listdir(folder):
        matrix = sio.loadmat(os.path.join(folder,file))['lms']
        if matrix is not None:
            matrices.append(matrix)
    return np.asarray(matrices)

def ld_images(folder):
    images=[]
    for file in os.listdir(folder):
        image = cv2.imread(os.path.join(folder,file))
        if image is not None:
            images.append(image)
    return np.asarray(images)

all_male_images_rgb = ld_images("male_images")
all_male_images_hsv = []
for i in range(len(all_male_images_rgb)):
    all_male_images_rgb[i] = cv2.cvtColor(all_male_images_rgb[i], cv2.COLOR_BGR2RGB)
    all_male_images_hsv.append(colors.rgb_to_hsv(all_male_images_rgb[i]))
all_male_images_hsv = np.array(all_male_images_hsv)
all_male_images = all_male_images_hsv[:, :, :, 2]
all_male_images = all_male_images.reshape(412,128*128)

all_female_images_rgb = ld_images("female_images")
all_female_images_hsv = []
for i in range(len(all_female_images_rgb)):
    all_female_images_rgb[i] = cv2.cvtColor(all_female_images_rgb[i], cv2.COLOR_BGR2RGB)
    all_female_images_hsv.append(colors.rgb_to_hsv(all_female_images_rgb[i]))
all_female_images_hsv = np.array(all_female_images_hsv)
all_female_images = all_female_images_hsv[:, :, :, 2]
all_female_images = all_female_images.reshape(588,128*128)

all_male_landmarks = ld_landmarks("male_landmarks")
all_male_landmarks = all_male_landmarks.reshape(412,68*2)

all_female_landmarks = ld_landmarks("female_landmarks")
all_female_landmarks = all_female_landmarks.reshape(588,68*2)

all_landmarks = np.append(all_male_landmarks, all_female_landmarks, axis=0)
mean_landmarks = np.mean(all_landmarks, axis=0)

# Remove this part of the code if you want non-warped images for the first question. Feed non-warped images into all_images instead 

warped_male_images = []
for i in range(len(all_male_images)):
    warped_male_images.append(mywarper.warp(all_male_images[i].reshape(128,128,1),all_male_landmarks[i].reshape(68,2),mean_landmarks.reshape(68,2)))
warped_male_images = np.array(warped_male_images)
warped_male_images = warped_male_images.reshape(412,128*128)

warped_female_images = []
for i in range(len(all_female_images)):
    warped_female_images.append(mywarper.warp(all_female_images[i].reshape(128,128,1),all_female_landmarks[i].reshape(68,2),mean_landmarks.reshape(68,2)))
warped_female_images = np.array(warped_female_images)
warped_female_images = warped_female_images.reshape(588,128*128)

###################################################################################################################################

all_images = np.append(warped_male_images, warped_female_images, axis=0)
mean_all_images = np.mean(all_images, axis=0)
all_images_labels = np.append(np.repeat(0,412), np.repeat(1,588))
all_landmarks = np.append(all_male_landmarks, all_female_landmarks, axis=0)
mean_landmarks = np.mean(all_landmarks, axis=0)

file_path = "aligned_eig_faces_all.npy"
if os.path.exists(file_path):
    all_eig_faces = np.load(file_path)
else:
    u, s, v = np.linalg.svd((all_images-mean_all_images).T)
    all_eig_faces = u
    np.save(file_path,all_eig_faces)
all_eig_faces = all_eig_faces.T
eig_faces = all_eig_faces[:50,]
all_pca_images = np.matmul((all_images - mean_all_images), eig_faces.T)

file_path = "eig_landmarks_all.npy"
if os.path.exists(file_path):
    all_eig_landmarks = np.load(file_path)
else:
    u, s, v = np.linalg.svd((all_landmarks-mean_landmarks).T)
    all_eig_landmarks = u
    np.save(file_path,all_eig_landmarks)
all_eig_landmarks = all_eig_landmarks.T
eig_landmarks = all_eig_landmarks[:10,]
pca_landmarks = np.matmul((all_landmarks - mean_landmarks), eig_landmarks.T)

pca_all_features = np.concatenate((all_pca_images, pca_landmarks), axis=1)

train_data, test_data, train_labels, test_labels = train_test_split(pca_all_features, all_images_labels, test_size = 200)

num_male = 0
num_female = 0
sum_male = np.zeros(60)
sum_female = np.zeros(60)
for i in range(len(train_data)):
    if(train_labels[i]==0):
        num_male += 1
        sum_male += train_data[i]
    else:
        num_female += 1
        sum_female += train_data[i]
mean_train_male = sum_male/num_male
mean_train_female = sum_female/num_female

s_f = 0
s_m = 0
for i in range(len(train_data)):
    if(train_labels[i]==0):
        s_m = s_m + np.matmul((train_data[i]-mean_train_male).reshape(60,1), ((train_data[i]-mean_train_male).reshape(60,1)).T)
    else:    
        s_f = s_f + np.matmul((train_data[i]-mean_train_female).reshape(60,1), ((train_data[i]-mean_train_female).reshape(60,1)).T)
s_w = s_m + s_f
fischer_face = np.matmul(np.linalg.inv(s_w), (mean_train_male - mean_train_female).reshape(60,1))
threshold = np.matmul((fischer_face.reshape(60,1)).T, (((mean_train_male + mean_train_female).reshape(60,1))/2))

result_labels = []
for i in range(len(test_data)):
    if(np.matmul((fischer_face.reshape(60,1)).T, test_data[i].reshape(60,1)) > threshold):
        result_labels.append(0)
    else:
        result_labels.append(1)
result_labels = np.array(result_labels)

score = 0
for i in range(len(result_labels)):
    if(test_labels[i] == result_labels[i]):
        score += 1
print("Test Accuracy : ", score/2)

s_f_app = 0
s_m_app = 0
for i in range(len(train_data)):
    if(train_labels[i]==0):
        s_m_app = s_m_app + np.matmul((train_data[i,:50]-mean_train_male[:50]).reshape(50,1), ((train_data[i,:50]-mean_train_male[:50]).reshape(50,1)).T)
    else:    
        s_f_app = s_f_app + np.matmul((train_data[i,:50]-mean_train_female[:50]).reshape(50,1), ((train_data[i,:50]-mean_train_female[:50]).reshape(50,1)).T)
s_w_app = s_m_app + s_f_app
fischer_face_app = np.matmul(np.linalg.inv(s_w_app), (mean_train_male[:50] - mean_train_female[:50]).reshape(50,1))
threshold_app = np.matmul((fischer_face_app.reshape(50,1)).T, (((mean_train_male[:50] + mean_train_female[:50]).reshape(50,1))/2))

s_f_lm = 0
s_m_lm = 0
for i in range(len(train_data)):
    if(train_labels[i]==0):
        s_m_lm = s_m_lm + np.matmul((train_data[i,50:60]-mean_train_male[50:60]).reshape(10,1), ((train_data[i,50:60]-mean_train_male[50:60]).reshape(10,1)).T)
    else:    
        s_f_lm = s_f_lm + np.matmul((train_data[i,50:60]-mean_train_female[50:60]).reshape(10,1), ((train_data[i,50:60]-mean_train_female[50:60]).reshape(10,1)).T)
s_w_lm = s_m_lm + s_f_lm
fischer_face_lm = np.matmul(np.linalg.inv(s_w_lm), (mean_train_male[50:60] - mean_train_female[50:60]).reshape(10,1))
threshold_lm = np.matmul((fischer_face_lm.reshape(10,1)).T, (((mean_train_male[50:60] + mean_train_female[50:60]).reshape(10,1))/2))

result_lm = []
for i in range(len(test_data)):
    result_lm.append(np.matmul((fischer_face_lm.reshape(10,1)).T, test_data[i,50:60].reshape(10,1)))
result_lm = np.array(result_lm)

result_app = []
for i in range(len(test_data)):
    result_app.append(np.matmul((fischer_face_app.reshape(50,1)).T, test_data[i,:50].reshape(50,1)))
result_app = np.array(result_app)

male_results_x = []
male_results_y = []
female_results_x = []
female_results_y = []
for i in range(len(test_labels)):
    if(test_labels[i] == 0):
        male_results_x.append(result_lm[i])
        male_results_y.append(result_app[i])
    else:
        female_results_x.append(result_lm[i])
        female_results_y.append(result_app[i])
male_results_x = np.array(male_results_x)
male_results_y = np.array(male_results_y)
female_results_x = np.array(female_results_x)
female_results_y = np.array(female_results_y)

plt.scatter(male_results_x, male_results_y, c='b', marker='+')
plt.scatter(female_results_x, female_results_y, c='r', marker='.')
x1 = 0
x2 = threshold_lm
y1 = threshold_app
y2 = 0
line_eqn = lambda x : ((y2-y1)/(x2-x1)) * (x - x1) + y1
xrange = np.arange(-0.005,0.005,0.001)
plt.plot(xrange.reshape(10), np.array([ line_eqn(x) for x in xrange]).reshape(10), color='y', linestyle='-', linewidth=2)
plt.xlabel("Landmarks")
plt.ylabel("Appearance")
plt.show()