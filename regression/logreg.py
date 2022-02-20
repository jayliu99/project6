# importing dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix # For computing accuracy

class BaseRegressor():
    def __init__(self, num_feats, learning_rate=0.1, tol=0.001, max_iter=100, batch_size=12):
        # initializing parameters
        self.W = np.random.randn(num_feats + 1).flatten()
        # assigning hyperparameters
        self.lr = learning_rate
        self.tol = tol
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.num_feats = num_feats
        # defining list for storing loss history
        self.loss_history_train = []
        self.loss_history_val = []
        
    def calculate_gradient(self, X, y):
        pass
    
    def loss_function(self, y_true, y_pred):
        pass
    
    def make_prediction(self, X):
        pass
    
    def train_model(self, X_train, y_train, X_val, y_val):
        # Padding data with vector of ones for bias term
        X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
        X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])

        # Defining intitial values for while loop
        prev_update_size = 1
        iteration = 1

        # Gradient descent
        while prev_update_size > self.tol and iteration < self.max_iter:

            # Shuffling the training data for each epoch of training
            shuffle_arr = np.concatenate([X_train, np.expand_dims(y_train, 1)], axis=1)

            # In place shuffle
            np.random.shuffle(shuffle_arr)
            X_train = shuffle_arr[:, :-1]
            y_train = shuffle_arr[:, -1].flatten()
            num_batches = int(X_train.shape[0]/self.batch_size) + 1
            X_batch = np.array_split(X_train, num_batches)
            y_batch = np.array_split(y_train, num_batches)

            # Generating list to save the param updates per batch
            update_size_epoch = []

            # Iterating through batches (full for loop is one epoch of training)
            for X_train, y_train in zip(X_batch, y_batch):
                # Making prediction on batch
                y_pred = self.make_prediction(X_train)

                # Calculating loss
                loss_train = self.loss_function(X_train, y_train)

                # Adding current loss to loss history record
                self.loss_history_train.append(loss_train)

                # Storing previous weights and bias
                prev_W = self.W

                # Calculating gradient of loss function with respect to each parameter
                grad = self.calculate_gradient(X_train, y_train)

                # Updating parameters
                new_W = prev_W - self.lr * grad 
                self.W = new_W

                # Saving step size
                update_size_epoch.append(np.abs(new_W - prev_W))

                # Validation pass
                loss_val = self.loss_function(X_val, y_val)
                self.loss_history_val.append(loss_val)

            # Defining step size as the average over the past epoch
            prev_update_size = np.mean(np.array(update_size_epoch))

            # Updating iteration number
            iteration += 1
    
    def plot_loss_history(self):
        """
        Plots the loss history after training is complete.
        """
        loss_hist = self.loss_history_train
        loss_hist_val = self.loss_history_val
        assert len(loss_hist) > 0, "Need to run training before plotting loss history"
        fig, axs = plt.subplots(2, figsize=(8,8))
        fig.suptitle('Loss History')
        axs[0].plot(np.arange(len(loss_hist)), loss_hist)
        axs[0].set_title('Training Loss')
        axs[1].plot(np.arange(len(loss_hist_val)), loss_hist_val)
        axs[1].set_title('Validation Loss')
        plt.xlabel('Steps')
        axs[0].set_ylabel('Train Loss')
        axs[1].set_ylabel('Val Loss')
        fig.tight_layout()
        plt.show()
        

# import required modules
class LogisticRegression(BaseRegressor):
    def __init__(self, num_feats, learning_rate=0.1, tol=0.0001, max_iter=100, batch_size=12):
        super().__init__(num_feats, learning_rate, tol, max_iter, batch_size)

    def set_W(self, W):
        """
        Set initial value of W (for unit testing purposes)

        Params:
            W (np.ndarray): weight vector with bias term (initialized)
        """

        self.W = W


    def get_W(self) -> np.ndarray:
        """
        Returns self.W for unit testing purposes

        Returns: 
            gradients for given loss function (np.ndarray)
        """
        return self.W

    def get_accuracy(self, X, y) -> float:
        """
        Returns the accuracy of predictions given true labels

        Params:
            X (np.ndarray): feature values
            y (np.array): labels corresponding to X

        Returns: 
            Accuracy of predictions on dataset X defined as (TN+TP)/(TN+TP+FN+FP)
        """
        y_pred = self.make_prediction(X)
        y_pred[y_pred>=0.5] = 1
        y_pred[y_pred<0.5] = 0
        cf = confusion_matrix(y, y_pred)

        # Compute accuracy of predictions
        accuracy = (cf[0,0] + cf[1,1]) / np.sum(cf)
        return accuracy


        
    def calculate_gradient(self, X, y) -> np.ndarray:
        """
        TODO: write function to calculate gradient of the
        logistic loss function to update the weights 

        Params:
            X (np.ndarray): feature values
            y (np.array): labels corresponding to X

        Returns: 
            gradients for given loss function (np.ndarray)
        """
        # X.shape = 1600 X 7
        # y.shape = 1600 X 1
        num_labels = y.shape[0]
        y_pred = self.make_prediction(X)               # y_pred.shape = 1600 X 1
        grad = (1/num_labels) * (X.T @ (y_pred - y))    # grad.shape = 7 X 1

        return grad
    
    def loss_function(self, X, y) -> float:
        """
        TODO: get y_pred from input X and implement binary cross 
        entropy loss function. Binary cross entropy loss assumes that 
        the classification is either 1 or 0, not continuous, making
        it more suited for (binary) classification.

        Params:
            X (np.ndarray): feature values
            y (np.array): labels corresponding to X

        Returns: 
            average loss 
        """
        # X.shape = 1600 X 7
        # y.shape = 1600 X 1

        y_pred = self.make_prediction(X)    # y_pred.shape = 1600 X 1
        y_0 = (1-y) * np.log(1-y_pred)      # Values where y=1 will be 0
        y_1 = y * np.log(y_pred)            # Values where y=0 will be 0
        loss = -1 * np.mean(y_0 + y_1, axis=0)

        return loss

    
    def make_prediction(self, X) -> np.array:
        """
        TODO: implement logistic function to get estimates (y_pred) for input
        X values. The logistic function is a transformation of the linear model W.T(X)+b 
        into an "S-shaped" curve that can be used for binary classification

        Params: 
            X (np.ndarray): Set of feature values to make predictions for

        Returns: 
            y_pred for given X
        """

        # X.shape = 1600 X 7
        # W.shape = 7 X 1

        # Addding bias term to X matrix if not already present
        if X.shape[1] == self.num_feats:
            X = np.hstack([X, np.ones((X.shape[0], 1))])

        raw_pred = -1 * (X @ self.W)     # raw_pred.shape = 1600 X 1
        exp_pred = np.exp(raw_pred)
        y_pred = 1/(1 + exp_pred)

        return y_pred.flatten()
        



    
