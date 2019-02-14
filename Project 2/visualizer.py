import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import cv2

class Visualizer:
	def __init__(self, histogram_intervals, top_wc_intervals):
		self.histogram_intervals = histogram_intervals
		self.top_wc_intervals = top_wc_intervals
		self.weak_classifier_accuracies = {}
		self.strong_classifier_scores = {}
		self.strong_classifier_errors = []
		self.filters = []
		self.labels = None
	
	def draw_histograms(self):
		for t in self.strong_classifier_scores:
			scores = self.strong_classifier_scores[t]
			pos_scores = [scores[idx] for idx, label in enumerate(self.labels) if label == 1]
			neg_scores = [scores[idx] for idx, label in enumerate(self.labels) if label == -1]

			bins = np.linspace(np.min(scores), np.max(scores), 100)

			plt.figure()
			plt.hist(pos_scores, bins, alpha=0.5, label='Faces')
			plt.hist(neg_scores, bins, alpha=0.5, label='Non-Faces')
			plt.legend(loc='upper right')
			plt.title('Using %d Weak Classifiers' % t)
			plt.savefig('histogram_%d.png' % t)

	def draw_rocs(self):
		plt.figure()
		for t in self.strong_classifier_scores:
			scores = self.strong_classifier_scores[t]
			fpr, tpr, _ = roc_curve(self.labels, scores)
			plt.plot(fpr, tpr, label = 'No. %d Weak Classifiers' % t)
		plt.legend(loc = 'lower right')
		plt.title('ROC Curve')
		plt.ylabel('True Positive Rate')
		plt.xlabel('False Positive Rate')
		plt.savefig('ROC Curve')

	def draw_sc_errors(self):
		plt.figure()
		errors = np.array(self.strong_classifier_errors)
		plt.plot(errors)
		plt.ylabel('Error')
		plt.xlabel('Number of Weak Classifiers')
		plt.title('Error of Strong Classifier using different number of Weak Classifiers')
		plt.savefig('Strong Classifier Errors')

	def draw_wc_accuracies(self):
			plt.figure()
			for t in self.weak_classifier_accuracies:
				plt.plot(self.weak_classifier_accuracies[t], label = 'After %d Selection' % t)
			plt.ylabel('Error')
			plt.xlabel('Weak Classifiers')
			plt.title('Top 1000 Weak Classifier Errors')
			plt.legend(loc = 'upper right')
			plt.savefig('Weak Classifier Errors')

	def draw_haar_filters(self):
		count = 1
		for f in self.filters:
			fig = np.full((16,16,3), 200).astype(np.uint8).copy()
			for p in f.plus_rects:
				cv2.rectangle(fig, (int(p[0]),int(p[1])), (int(p[2]),int(p[3])), (255, 255, 255) if f.polarity == 1 else (0,0,0), -1)
			for m in f.minus_rects:
				cv2.rectangle(fig, (int(m[0]),int(m[1])), (int(m[2]),int(m[3])), (0, 0, 0) if(f.polarity == 1) else (255, 255, 255), -1)
			plt.figure()
			plt.imshow(fig)
			plt.gca().invert_yaxis()
			plt.title('Haar Filter No. '+ str(count))
			plt.savefig('Top Haar Filter No. ' + str(count) +'.png')
			plt.close()
			count += 1

if __name__ == '__main__':
	main()
