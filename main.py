import numpy as np
import pandas as pd
from regression import (logreg, utils)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

def main():

    # load data with default settings
    X_train, X_val, y_train, y_val = utils.loadDataset(features=['Penicillin V Potassium 500 MG', 'Computed tomography of chest and abdomen', 
                                    'Plain chest X-ray (procedure)',  'Low Density Lipoprotein Cholesterol', 
                                    'Creatinine', 'AGE_DIAGNOSIS'], split_percent=0.8, split_state=42)

    # scale data since values vary across features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform (X_val)
    # # W_test = np.random.randn(6).flatten()
    # # y_test = -1 * X_train @ W_test
    # # y_test = 1/(1 + y_test)
    # # print(y_test.shape)
    # # print(W_test.shape)
    # print(X_train.shape, X_val.shape, y_val.shape, y_train.shape)

    
    # # for testing purposes once you've added your code
    # # CAUTION: hyperparameters have not been optimized

    log_model = logreg.LogisticRegression(num_feats=6, max_iter=100, tol=0.001, learning_rate=0.4, batch_size=50)

    # num_labels = y_train.shape[0]
    # y_pred = log_model.make_prediction(X_train)               # y_pred.shape = 1600 X 1
    # grad = (-1/num_labels * X_train).T @ (y_pred - y_train)    # grad.shape = 7 X 1
    # print(y_pred - y_train)

    log_model.train_model(X_train, y_train, X_val, y_val)
    log_model.plot_loss_history()

    loss_train = log_model.loss_function(X_train, y_train)
    loss_val = log_model.loss_function(X_val, y_val)

    print(loss_train, loss_val)

    # y_pred_train = log_model.make_prediction(X_train)
    # y_pred_train[y_pred_train>=0.5] = 1
    # y_pred_train[y_pred_train<0.5] = 0
    # train_cf = confusion_matrix(y_train, y_pred_train)

    # y_pred_val = log_model.make_prediction(X_val)
    # y_pred_val[y_pred_val>=0.5] = 1
    # y_pred_val[y_pred_val<0.5] = 0
    # val_cf = confusion_matrix(y_val, y_pred_val)

    # accuracy_train = (train_cf[0,0] + train_cf[1,1]) / np.sum(train_cf)
    # accuracy_val = (val_cf[0,0] + val_cf[1,1]) / np.sum(val_cf)

    # print("Accuracy for training:", accuracy_train)
    # print("Accuracy for validation:", accuracy_val)

    # Test y_pred computation using a smaller dataset

    # X = np.array([[1, 2, 3],
    #                [2, 3, 1],
    #                 [1, 3, 2]])
    # W = np.array([2, 2, 2])

    # raw_pred = -1 * X @ W      # raw_pred.shape = 1600 X 1
    # exp_pred = np.exp(raw_pred)
    # y_pred = 1/(1 + exp_pred)
    # #print(y_pred)
    # # #y_pred[y_pred>=0.5] = 1
    # # #y_pred[y_pred<0.5] = 0
    # y_pred.flatten()
    # print(y_pred.shape)
    # print(y_pred)
            
    

if __name__ == "__main__":
    main()
