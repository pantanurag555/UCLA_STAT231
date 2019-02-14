import os
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import pickle
import time

import cv2
from weak_classifier import Ada_Weak_Classifier, Real_Weak_Classifier
from im_process import image2patches, nms, normalize
from copy import deepcopy


class Boosting_Classifier:
	def __init__(self, haar_filters, data, labels, num_chosen_wc, num_bins, visualizer, num_cores, style):
		self.filters = haar_filters
		self.data = data
		self.labels = labels
		self.num_chosen_wc = num_chosen_wc
		self.num_bins = num_bins
		self.visualizer = visualizer
		self.num_cores = num_cores
		self.style = style
		self.chosen_wcs = None
		if style == 'Ada':
			self.weak_classifiers = [Ada_Weak_Classifier(i, filt[0], filt[1], self.num_bins)\
									 for i, filt in enumerate(self.filters)]
		elif style == 'Real':
			self.weak_classifiers = [Real_Weak_Classifier(i, filt[0], filt[1], self.num_bins)\
									 for i, filt in enumerate(self.filters)]
	
	def calculate_training_activations(self, save_dir = None, load_dir = None):
		print('Calcuate activations for %d weak classifiers, using %d images.' % (len(self.weak_classifiers), self.data.shape[0]))
		if load_dir is not None and os.path.exists(load_dir):
			print('[Find cached activations, %s loading...]' % load_dir)
			wc_activations = np.load(load_dir)
		else:
			if self.num_cores == 1:
				wc_activations = [wc.apply_filter(self.data) for wc in self.weak_classifiers]
			else:
				wc_activations = Parallel(n_jobs = self.num_cores)(delayed(wc.apply_filter)(self.data) for wc in self.weak_classifiers)
			wc_activations = np.array(wc_activations)
			if save_dir is not None:
				print('Writing results to disk...')
				np.save(save_dir, wc_activations)
				print('[Saved calculated activations to %s]' % save_dir)
		for wc in self.weak_classifiers:
			wc.activations = wc_activations[wc.id, :]
		return wc_activations
	
	#select weak classifiers to form a strong classifier
	#after training, by calling self.sc_function(), a prediction can be made
	#self.chosen_wcs should be assigned a value after self.train() finishes
	#call Weak_Classifier.calc_error() in this function
	#cache training results to self.visualizer for visualization
	#
	#
	#detailed implementation is up to you
	#consider caching partial results and using parallel computing
	def train(self, save_dir = None):
		if(self.style == 'Ada'):
			if(save_dir is not None and os.path.exists(save_dir)):
				self.chosen_wcs = pickle.load(open(save_dir, 'rb'))
				return
			data_weight = np.full((self.data.shape[0]), 1 / self.data.shape[0])
			self.chosen_wcs = []
			for i in range(self.num_chosen_wc):
				start = time.clock()
				err_th_po = Parallel(n_jobs=self.num_cores)(delayed(self.weak_classifiers[j].calc_error)(data_weight, self.labels) for j in range(len(self.weak_classifiers)))
				err_th_po = np.array(err_th_po)
				index = np.argmin(err_th_po[:,0])
				min_error = err_th_po[index,0]
				self.weak_classifiers[index].threshold = err_th_po[index,1]
				self.weak_classifiers[index].polarity = err_th_po[index,2]
				alpha = 0.5 * np.log( (1 - min_error) / min_error )
				self.chosen_wcs.append( (alpha, deepcopy(self.weak_classifiers[index])) )		
				train_predicts = []
				for idx in range(self.data.shape[0]):
					train_predicts.append(self.sc_function(self.data[idx, ...]))
				scores = train_predicts
				if(i == 0 or i == 9 or i == 49 or i == 99):
					errors = np.array(err_th_po[:,0])
					best_errors = np.sort(errors)
					self.visualizer.weak_classifier_accuracies[i+1] = best_errors[0:999]
					self.visualizer.strong_classifier_scores[i+1] = scores
				self.visualizer.strong_classifier_errors.append(1 - np.mean(np.sign(scores) == self.labels))
				final_pred = np.array([self.weak_classifiers[index].predict_image(self.data[j]) for j in range(len(self.data))])
				end = time.clock()
				print("No. ",i," Index ",index," Error ",min_error," Alpha ",alpha," Time ",end-start)
				data_weight = np.array(data_weight * np.exp(-1 * self.labels * alpha * final_pred))
				data_weight = np.array(data_weight / np.sum(data_weight))
			self.chosen_wcs = np.array(self.chosen_wcs)
			if save_dir is not None:
				pickle.dump(self.chosen_wcs, open(save_dir, 'wb'))
		else:
			ada_chosen_wcs = pickle.load(open(save_dir, 'rb'))
			indices = [ada_chosen_wcs[j,1].id for j in range(len(ada_chosen_wcs))]
			data_weight = np.full((self.data.shape[0]), 1 / self.data.shape[0])
			alpha = 1
			self.chosen_wcs = []
			cnt = 0
			for i in indices:
				self.weak_classifiers[i].calc_error(data_weight, self.labels)
				self.chosen_wcs.append( (alpha, deepcopy(self.weak_classifiers[i])) )
				train_predicts = []
				for idx in range(self.data.shape[0]):
					train_predicts.append(self.sc_function(self.data[idx, ...]))
				scores = train_predicts
				print("Index ",cnt)
				if(cnt == 9 or cnt == 49 or cnt == 99):
					self.visualizer.strong_classifier_scores[cnt+1] = scores
				final_pred = np.array([self.weak_classifiers[i].predict_image(self.data[j]) for j in range(len(self.data))])
				data_weight = np.array(data_weight * np.exp(-1 * self.labels * alpha * final_pred))
				data_weight = np.array(data_weight / np.sum(data_weight))
				cnt += 1

	def sc_function(self, image):
		return np.sum([np.array([alpha * wc.predict_image(image) for alpha, wc in self.chosen_wcs])])			

	def load_trained_wcs(self, save_dir):
		self.chosen_wcs = pickle.load(open(save_dir, 'rb'))	

	def face_detection(self, img, rgb_img, scale_step = 20):
		
		# this training accuracy should be the same as your training process,
		##################################################################################
		train_predicts = []
		for idx in range(self.data.shape[0]):
			train_predicts.append(self.sc_function(self.data[idx, ...]))
		print('Check training accuracy is: ', np.mean(np.sign(train_predicts) == self.labels))
		##################################################################################

		scales = 1 / np.linspace(1, 8, scale_step)
		patches, patch_xyxy = image2patches(scales, img)
		print('Face Detection in Progress ..., total %d patches' % patches.shape[0])
		predicts = [self.sc_function(patch) for patch in tqdm(patches)]
		print(np.mean(np.array(predicts) > 0), np.sum(np.array(predicts) > 0))
		pos_predicts_xyxy = np.array([patch_xyxy[idx] + [score] for idx, score in enumerate(predicts) if score > 0])
		if pos_predicts_xyxy.shape[0] == 0:
			return
		xyxy_after_nms = nms(pos_predicts_xyxy, 0.01)
		
		print('after nms:', xyxy_after_nms.shape[0])
		for idx in range(xyxy_after_nms.shape[0]):
			pred = xyxy_after_nms[idx, :]
			cv2.rectangle(rgb_img, (int(pred[0]), int(pred[1])), (int(pred[2]), int(pred[3])), (0, 255, 0), 2) #green rectangular with line width 3
		return rgb_img

	def get_hard_negative_patches(self, img, scale_step = 10):
		scales = 1 / np.linspace(1, 8, scale_step)
		patches, patch_xyxy = image2patches(scales, img)
		print('Get Hard Negative in Progress ..., total %d patches' % patches.shape[0])
		predicts = [self.sc_function(patch) for patch in tqdm(patches)]
		predicts = np.array(predicts)
		print(predicts.shape)
		wrong_patches = patches[np.where(predicts > 0), ...]
		wrong_patches = np.array(wrong_patches)
		wrong_patches = np.squeeze(wrong_patches)
		return wrong_patches

	def visualize(self):
		for i in range(20):
			self.visualizer.filters.append(self.chosen_wcs[i][1])
		self.visualizer.labels = self.labels
		self.visualizer.draw_haar_filters()
		self.visualizer.draw_sc_errors()
		self.visualizer.draw_histograms()
		self.visualizer.draw_rocs()
		self.visualizer.draw_wc_accuracies()

	def real_visualize(self):
		self.visualizer.labels = self.labels
		self.visualizer.draw_histograms()
		self.visualizer.draw_rocs()
