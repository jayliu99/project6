"""
Write your logreg unit tests here. Some examples of tests we will be looking for include:
* check that fit appropriately trains model & weights get updated
* check that predict is working
* check that loss approaches 0

More details on potential tests below, these are not exhaustive
"""
import numpy as np
import pandas as pd
from regression import (logreg, utils)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

def test_updates():

	# load data with default settings
	X_train, X_val, y_train, y_val = utils.loadDataset(features=['Penicillin V Potassium 500 MG', 'Computed tomography of chest and abdomen', 
									'Plain chest X-ray (procedure)',  'Low Density Lipoprotein Cholesterol', 
									'Creatinine', 'AGE_DIAGNOSIS'], split_percent=0.8, split_state=42)

	# scale data since values vary across features
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_val = sc.transform (X_val)

	# Load logistic regression model
	log_model = logreg.LogisticRegression(num_feats=6, max_iter=100, tol=0.001, learning_rate=0.4, batch_size=50)

	# Train model & check that your gradient is being calculated correctly
	start_grad = log_model.calculate_gradient(X_train, y_train)
	log_model.train_model(X_train, y_train, X_val, y_val)
	end_grad = log_model.calculate_gradient(X_train, y_train)
	# At the end of training, gradient should be less negative (closer to zero) than it was at the start
	assert(np.sum(end_grad) > np.sum(start_grad))

	# Check that your loss function is correct and that 
	# you have reasonable losses at the end of training
	loss_train = log_model.loss_function(X_train, y_train)
	loss_val = log_model.loss_function(X_val, y_val)
	assert(loss_train < 0.5)
	assert(loss_val < 0.5)


def test_predict():

	# load data with default settings
	X_train, X_val, y_train, y_val = utils.loadDataset(features=['Penicillin V Potassium 500 MG', 'Computed tomography of chest and abdomen', 
									'Plain chest X-ray (procedure)',  'Low Density Lipoprotein Cholesterol', 
									'Creatinine', 'AGE_DIAGNOSIS'], split_percent=0.8, split_state=42)

	# scale data since values vary across features
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_val = sc.transform (X_val)

	# Load logistic regression model
	log_model = logreg.LogisticRegression(num_feats=6, max_iter=100, tol=0.001, learning_rate=0.4, batch_size=50)

	# Train model & check that self.W is being updated as expected
	start_W = log_model.get_W()
	log_model.train_model(X_train, y_train, X_val, y_val)
	end_W = log_model.get_W()
	# Check that at the end of training, the weights have been updated
	assert(not np.array_equal(start_W, end_W))

	# Check that model produces reasonable estimates for NSCLC classification
	accuracy_train = log_model.get_accuracy(X_train, y_train)
	accuracy_val = log_model.get_accuracy(X_val, y_val)
	# "Reasonable" estimates = accuracy of at least 70%
	assert(accuracy_train > 0.7)
	assert(accuracy_val > 0.7)

	





