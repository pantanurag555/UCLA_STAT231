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

########################## COMMON FUNCTIONS (Used in all the parts)  ##########################

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

def plot_error(test_images, K, step, eig_faces, mean_image, val):
    x = []
    y = []
    i = 1
    while i<=K:
        tmp_eig_faces = eig_faces[:i,]
        pca_faces = np.matmul((test_images - mean_image), tmp_eig_faces.T)
        recon_faces = (np.matmul(pca_faces, tmp_eig_faces) + mean_image)
        error = np.mean(((recon_faces - test_images) / val) ** 2)
        x.append(i)
        y.append(error)
        if(i==1 and step>1):
            i -= 1
        i += step
    plt.scatter(x,y)
    plt.plot(x,y)
    plt.show()

def plot_error_hsv(all_images, test_images, test_landmarks, mean_image, mean_landmarks, eig_faces, eig_landmarks, K, step):
    x = []
    y = []
    test_images_original = all_images[800:]
    p = 1
    while p<=K:
        tmp_eig_faces = eig_faces[:p,]
        pca_faces = np.matmul((test_images - mean_image), tmp_eig_faces.T)
        pca_landmarks = np.matmul((test_landmarks - mean_landmarks), eig_landmarks.T)
        recon_faces = np.matmul(pca_faces, tmp_eig_faces) + mean_image
        recon_landmarks = np.matmul(pca_landmarks, eig_landmarks) + mean_landmarks
        final_recon_faces = []
        for i in range(len(recon_faces)):
            final_recon_faces.append(mywarper.warp(recon_faces[i].reshape(128,128,1),mean_landmarks.reshape(68,2),recon_landmarks[i].reshape(68,2)))
        final_recon_faces = np.array(final_recon_faces)
        final_recon_faces = final_recon_faces.reshape(200,128*128)
        error = np.mean(((final_recon_faces - test_images_original) / 255) ** 2)
        x.append(p)
        y.append(error)
        if(p==1 and step>1):
            p -= 1
        p += step
    plt.scatter(x,y)
    plt.plot(x,y)
    plt.show()

############# PART 1 (Execute as standalone, after commenting out the other parts)  #############

all_images_rgb = ld_images("images")
all_images_hsv = []
for i in range(len(all_images_rgb)):
    all_images_rgb[i] = cv2.cvtColor(all_images_rgb[i], cv2.COLOR_BGR2RGB)
    all_images_hsv.append(colors.rgb_to_hsv(all_images_rgb[i]))
all_images_hsv = np.array(all_images_hsv)
all_images = all_images_hsv[:, :, :, 2]
all_images = all_images.reshape(1000,128*128)
train_images = all_images[:800,]
test_images = all_images[800:,]
mean_image = np.mean(train_images, axis = 0)
file_path = "eig_faces.npy"
if os.path.exists(file_path):
    all_eig_faces = np.load(file_path)
else:
    u, s, v = np.linalg.svd((train_images-mean_image).T)
    all_eig_faces = u
    np.save(file_path,all_eig_faces)
all_eig_faces = all_eig_faces.T
eig_faces = all_eig_faces[:50,]
for image in eig_faces[:10,]:
    plt.imshow(image.reshape(128,128))
    plt.show()
pca_faces = np.matmul((test_images - mean_image), eig_faces.T)
recon_faces = np.matmul(pca_faces, eig_faces) + mean_image
print(np.mean(((recon_faces/255)-(test_images/255))**2))
test_images_rgb = all_images_hsv[800:,:,:,:]
recon_faces = recon_faces.reshape(200,128,128)
tmp = np.zeros((200,128,128,3))
tmp[:,:,:,0] = all_images_hsv[800:,:,:,0]
tmp[:,:,:,1] = all_images_hsv[800:,:,:,1]
tmp[:,:,:,2] = recon_faces
recon_faces_rgb = tmp
for i in range(len(recon_faces_rgb)):
    recon_faces_rgb[i] = colors.hsv_to_rgb(recon_faces_rgb[i])
for i in range(len(test_images_rgb)):
    test_images_rgb[i] = colors.hsv_to_rgb(test_images_rgb[i])
recon_faces_rgb = recon_faces_rgb.astype(int)
test_images_rgb = test_images_rgb.astype(int)
rows = 1
columns = 2
for i in range(10):
    fig = plt.figure(figsize=(7,7))
    fig_plot_index = 1
    fig.add_subplot(rows, columns, fig_plot_index)
    plt.imshow(test_images_rgb[i])
    fig_plot_index += 1
    fig.add_subplot(rows, columns, fig_plot_index)
    plt.imshow(recon_faces_rgb[i])
    plt.show()
plot_error(test_images, 50, 5, eig_faces, mean_image, 255)

############# PART 2 (Execute as standalone, after commenting out the other parts)  #############

all_landmarks = ld_landmarks("landmarks")
all_landmarks = all_landmarks.reshape(1000,68*2)
train_landmarks = all_landmarks[:800,]
test_landmarks = all_landmarks[800:,]
mean_landmarks = np.mean(train_landmarks, axis = 0)
file_path = "eig_landmarks.npy"
if os.path.exists(file_path):
    all_eig_landmarks = np.load(file_path)
else:
    u, s, v = np.linalg.svd((train_landmarks-mean_landmarks).T)
    all_eig_landmarks = u
    np.save(file_path,all_eig_landmarks)
all_eig_landmarks = all_eig_landmarks.T
eig_landmarks = all_eig_landmarks[:10,]
pca_landmarks = np.matmul((test_landmarks - mean_landmarks), eig_landmarks.T)
recon_landmarks = np.matmul(pca_landmarks, eig_landmarks) + mean_landmarks
fig,ax = plt.subplots()
for i in range(10):
    tmp = (eig_landmarks[i] + mean_landmarks).reshape(68,2)
    ax.scatter(tmp[:,0],tmp[:,1])
plt.gca().invert_yaxis()
plt.show()
plot_error(test_landmarks, 10, 1, eig_landmarks, mean_landmarks, 128)

############# PART 3 (Execute as standalone, after commenting out the other parts)  #############

all_images_rgb = ld_images("images")
all_images_hsv = []
for i in range(len(all_images_rgb)):
    all_images_rgb[i] = cv2.cvtColor(all_images_rgb[i], cv2.COLOR_BGR2RGB)
    all_images_hsv.append(colors.rgb_to_hsv(all_images_rgb[i]))
all_images_hsv = np.array(all_images_hsv)
all_images = all_images_hsv[:, :, :, 2]
all_images = all_images.reshape(1000,128*128)
all_landmarks = ld_landmarks("landmarks")
all_landmarks = all_landmarks.reshape(1000,68*2)
train_landmarks = all_landmarks[:800,]
test_landmarks = all_landmarks[800:,]
mean_landmarks = np.mean(train_landmarks, axis = 0)
file_path = "eig_landmarks.npy"
if os.path.exists(file_path):
    all_eig_landmarks = np.load(file_path)
else:
    u, s, v = np.linalg.svd((train_landmarks-mean_landmarks).T)
    all_eig_landmarks = u
    np.save(file_path,all_eig_landmarks)
all_eig_landmarks = all_eig_landmarks.T
eig_landmarks = all_eig_landmarks[:10,]
warp_all_images = []
for i in range(len(all_images)):
    warp_all_images.append(mywarper.warp(all_images[i].reshape(128,128,1),all_landmarks[i].reshape(68,2),mean_landmarks.reshape(68,2)))
warp_all_images = np.array(warp_all_images)
warp_all_images = warp_all_images.reshape(1000,128*128)
train_images = warp_all_images[:800,]
test_images = warp_all_images[800:,]
mean_image = np.mean(train_images, axis = 0)
file_path = "aligned_eig_faces.npy"
if os.path.exists(file_path):
    all_eig_faces = np.load(file_path)
else:
    u, s, v = np.linalg.svd((train_images-mean_image).T)
    all_eig_faces = u
    np.save(file_path,all_eig_faces)
all_eig_faces = all_eig_faces.T
eig_faces = all_eig_faces[:50,]
pca_faces = np.matmul((test_images - mean_image), eig_faces.T)
pca_landmarks = np.matmul((test_landmarks - mean_landmarks), eig_landmarks.T)
recon_faces = np.matmul(pca_faces, eig_faces) + mean_image
recon_landmarks = np.matmul(pca_landmarks, eig_landmarks) + mean_landmarks
final_recon_faces = []
for i in range(len(recon_faces)):
    final_recon_faces.append(mywarper.warp(recon_faces[i].reshape(128,128,1),mean_landmarks.reshape(68,2),recon_landmarks[i].reshape(68,2)))
final_recon_faces = np.array(final_recon_faces)
final_recon_faces = final_recon_faces.reshape(200,128,128)
test_images_rgb = all_images_hsv[800:,:,:,:]
tmp = np.zeros((200,128,128,3))
tmp[:,:,:,0] = all_images_hsv[800:,:,:,0]
tmp[:,:,:,1] = all_images_hsv[800:,:,:,1]
tmp[:,:,:,2] = final_recon_faces
final_recon_faces_rgb = tmp
for i in range(len(final_recon_faces_rgb)):
    final_recon_faces_rgb[i] = colors.hsv_to_rgb(final_recon_faces_rgb[i])
for i in range(len(test_images_rgb)):
    test_images_rgb[i] = colors.hsv_to_rgb(test_images_rgb[i])
final_recon_faces_rgb = final_recon_faces_rgb.astype(int)
test_images_rgb = test_images_rgb.astype(int)
rows = 1
columns = 2
for i in range(20):
    fig = plt.figure(figsize=(7,7))
    fig_plot_index = 1
    fig.add_subplot(rows, columns, fig_plot_index)
    plt.imshow(test_images_rgb[i])
    fig_plot_index += 1
    fig.add_subplot(rows, columns, fig_plot_index)
    plt.imshow(final_recon_faces_rgb[i])
    plt.show()
plot_error_hsv(all_images, test_images, test_landmarks, mean_image, mean_landmarks, eig_faces, eig_landmarks, 50, 5 )

############# PART 4 (Execute as standalone, after commenting out the other parts)  #############

all_images_rgb = ld_images("images")
all_images_hsv = []
for i in range(len(all_images_rgb)):
    all_images_rgb[i] = cv2.cvtColor(all_images_rgb[i], cv2.COLOR_BGR2RGB)
    all_images_hsv.append(colors.rgb_to_hsv(all_images_rgb[i]))
all_images_hsv = np.array(all_images_hsv)
all_images = all_images_hsv[:, :, :, 2]
all_images = all_images.reshape(1000,128*128)
all_landmarks = ld_landmarks("landmarks")
all_landmarks = all_landmarks.reshape(1000,68*2)
train_landmarks = all_landmarks[:800,]
test_landmarks = all_landmarks[800:,]
mean_landmarks = np.mean(train_landmarks, axis = 0)
file_path = "eig_landmarks.npz"
if os.path.exists(file_path):
    all_eig_landmarks_values = np.load(file_path)
    all_eig_landmarks  = all_eig_landmarks_values['arr_0']
    all_eig_lvalues  = all_eig_landmarks_values['arr_1']
else:
    u, s, v = np.linalg.svd((train_landmarks-mean_landmarks).T)
    np.savez(file_path, u, s)
    all_eig_landmarks = u
    all_eig_lvalues = s
all_eig_landmarks = all_eig_landmarks.T
eig_landmarks = all_eig_landmarks[:10,]
eig_lvalues = all_eig_lvalues[:10,]
warp_all_images = []
for i in range(len(all_images)):
    warp_all_images.append(mywarper.warp(all_images[i].reshape(128,128,1),all_landmarks[i].reshape(68,2),mean_landmarks.reshape(68,2)))
warp_all_images = np.array(warp_all_images)
warp_all_images = warp_all_images.reshape(1000,128*128)
train_images = warp_all_images[:800,]
test_images = warp_all_images[800:,]
mean_image = np.mean(train_images, axis = 0)
file_path = "aligned_eig_faces.npz"
if os.path.exists(file_path):
    all_eig_faces_values = np.load(file_path)
    all_eig_faces  = all_eig_faces_values['arr_0']
    all_eig_fvalues  = all_eig_faces_values['arr_1']
else:
    u, s, v = np.linalg.svd((train_images-mean_image).T)
    np.savez(file_path, u, s)
    all_eig_faces = u
    all_eig_fvalues = s
all_eig_faces = all_eig_faces.T
eig_faces = all_eig_faces[:50,]
eig_fvalues = all_eig_fvalues[:50,]
new_images_pca = []
for i in range(50):
    tmp = np.zeros(50)
    for j in range(50):
        tmp[j] = np.random.normal(0,np.sqrt(eig_fvalues[j]))
    new_images_pca.append(tmp)
new_images_pca = np.array(new_images_pca)
new_images = np.matmul(new_images_pca, eig_faces) + mean_image
new_landmarks_pca = []
for i in range(50):
    tmp = np.zeros(10)
    for j in range(10):
        tmp[j] = np.random.normal(0,np.sqrt(eig_lvalues[j]))
    new_landmarks_pca.append(tmp)
new_landmarks_pca = np.array(new_landmarks_pca)
new_landmarks = np.matmul(new_landmarks_pca, eig_landmarks) + mean_landmarks
warped_new_images = []
for i in range(len(new_images)):
    warped_new_images.append(mywarper.warp(new_images[i].reshape(128,128,1),mean_landmarks.reshape(68,2),new_landmarks[i].reshape(68,2)))
warped_new_images = np.array(warped_new_images).reshape(50,128,128)
for i in range(5):
    fig_plot_index = 0
    fig,axes = plt.subplots(nrows=1,ncols=10, sharey=True, figsize=(20,4))
    for img,ax in zip(warped_new_images[i*10:],axes):
        ax.imshow(img, cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()