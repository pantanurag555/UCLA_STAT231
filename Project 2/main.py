import numpy as np
import time
import cv2
import pickle
from boosting_classifier import Boosting_Classifier
from visualizer import Visualizer
from im_process import normalize
from utils import *

def main():
	#flag for debugging
	flag_subset = False
	boosting_type = 'Ada' #'Real' or 'Ada'
	training_epochs = 100 if not flag_subset else 20
	act_cache_dir = 'wc_activations.npy' if not flag_subset else 'wc_activations_subset.npy'
	chosen_wc_cache_dir = 'chosen_wcs.pkl' if not flag_subset else 'chosen_wcs_subset.pkl'

	#data configurations
	pos_data_dir = 'newface16'
	neg_data_dir = 'nonface16'
	image_w = 16
	image_h = 16
	data, labels = load_data(pos_data_dir, neg_data_dir, image_w, image_h, flag_subset)
	data = integrate_images(normalize(data))
	#number of bins for boosting
	num_bins = 25

	#number of cpus for parallel computing
	num_cores = 6 if not flag_subset else 1 #always use 1 when debugging
	
	#create Haar filters
	filters = generate_Haar_filters(4, 4, 16, 16, image_w, image_h, flag_subset)

	#create visualizer to draw histograms, roc curves and best weak classifier accuracies
	drawer = Visualizer([10, 20, 50, 100], [1, 10, 20, 50, 100])
	
	#create boost classifier with a pool of weak classifier
	boost = Boosting_Classifier(filters, data, labels, training_epochs, num_bins, drawer, num_cores, boosting_type)

	#calculate filter values for all training images
	start = time.clock()
	boost.calculate_training_activations(act_cache_dir, act_cache_dir)
	end = time.clock()
	print('%f seconds for activation calculation' % (end - start))

	boost.train(chosen_wc_cache_dir)

	if(boosting_type == 'Ada'):	
		boost.visualize()
		original_img = cv2.imread('./Testing_Images/Face_4.jpg', cv2.IMREAD_GRAYSCALE)
		original_rgb_img = cv2.imread('./Testing_Images/Face_4.jpg')
		result_img = boost.face_detection(original_img, original_rgb_img)
		cv2.imwrite('Result_img_%s.png' % boosting_type, result_img)
		save_hard_neg = 'hard_neg.pkl'
		save_hard_neg_labels = 'hard_neg_label.pkl'
		if(os.path.exists(save_hard_neg)):
			final_hard_negatives = pickle.load(open(save_hard_neg, 'rb'))
			final_neg_labels = pickle.load(open(save_hard_neg_labels, 'rb'))
		else:
			for i in range(3):
				hard_negative_img = cv2.imread('./Testing_Images/Non_face_' + str(i+1) + '.jpg', cv2.IMREAD_GRAYSCALE)
				hard_negatives = boost.get_hard_negative_patches(hard_negative_img)
				print("No. of hard negative patches: ", hard_negatives.shape)
				hard_neg_labels = np.full((hard_negatives.shape[0]), -1)
				if(i == 0):
					final_hard_negatives = hard_negatives
					final_neg_labels = hard_neg_labels
				else:
					final_hard_negatives = np.append(final_hard_negatives, hard_negatives, axis = 0)
					final_neg_labels = np.append(final_neg_labels, hard_neg_labels, axis = 0)
			pickle.dump(final_hard_negatives, open(save_hard_neg, 'wb'))
			pickle.dump(final_neg_labels, open(save_hard_neg_labels, 'wb'))
		boost.data = np.append(boost.data, final_hard_negatives, axis = 0)
		boost.labels = np.append(boost.labels, final_neg_labels, axis = 0)
		new_act_cache_dir = 'new_wc_activations.npy' if not flag_subset else 'new_wc_activations_subset.npy'
		new_chosen_wc_cache_dir = 'new_chosen_wcs.pkl' if not flag_subset else 'new_chosen_wcs_subset.pkl'
		start = time.clock()
		boost.calculate_training_activations(new_act_cache_dir, new_act_cache_dir)
		end = time.clock()
		boost.train(new_chosen_wc_cache_dir)
		original_img = cv2.imread('./Testing_Images/Face_4.jpg', cv2.IMREAD_GRAYSCALE)
		original_rgb_img = cv2.imread('./Testing_Images/Face_4.jpg')
		result_img = boost.face_detection(original_img, original_rgb_img)
		cv2.imwrite('New_result_img_%s.png' % boosting_type, result_img)
	else:
		boost.real_visualize()

if __name__ == '__main__':
	main()
