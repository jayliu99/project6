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
    # print(X_train.shape, X_val.shape, y_val.shape, y_train.shape)

    
    # # for testing purposes once you've added your code
    # # CAUTION: hyperparameters have not been optimized

    log_model = logreg.LogisticRegression(num_feats=6, max_iter=100, tol=0.001, learning_rate=0.4, batch_size=50)

    log_model.set_W(np.array([0, 0, 0, 0, 0, 0, 0]))

    log_model.train_model(X_train, y_train, X_val, y_val)
    log_model.plot_loss_history()


    print(log_model.get_W())
    print(log_model.calculate_gradient(X_train, y_train))
    print(log_model.get_accuracy(X_train, y_train))
    print(log_model.get_accuracy(X_val, y_val))

    loss_train = log_model.loss_function(X_train, y_train)
    loss_val = log_model.loss_function(X_val, y_val)

    print(loss_train, loss_val)

    

if __name__ == "__main__":
    main()
