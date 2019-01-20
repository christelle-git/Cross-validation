"""

            
             Implementation of the cross validation
          
          
   Procedure:
      * cut the training set in folds using StratifiedKFolds.
      * on each fold apply the kNN model with KNeighborsClassifier().
      * calculate the arithmetic average score for each splitting.
      * determine the best performance on score accuracy.
                         
   Args:  
      * list of hyperparameters: l_neighbors = [3, 5, 7, 9, 11, 13, 15]
        (number of neighbors for the kNN algorithm)
      * number of folds: nbFolds = 5

"""
from sklearn.model_selection import StratifiedKFold
from sklearn import neighbors, metrics


# MY GRID SEARCH
def my_cross_val(l_neighbors, nbFolds, X, y, Xtest, ytest):
    best_i, best_score = -1, -1
    clfs = {}
    # cut in nbFolds parts [0, 1, ..., n-1] of size (n/k)
    skf = StratifiedKFold(n_splits = nbFolds)
    for i in l_neighbors:
        clf_kf = neighbors.KNeighborsClassifier(n_neighbors=i)
        k = 1
        som = 0
        for train_index, test_index in skf.split(X, y):
            X_train_kfold, X_test_kfold = X[train_index], X[test_index]
            y_train_kfold, y_test_kfold = y[train_index], y[test_index]
            # optimize classifier on the training set
            clf_kf.fit(X_train_kfold, y_train_kfold)
            
            som = som + clf_kf.score(X_test_kfold, y_test_kfold)   
            k = k+1
    
        moy = som/nbFolds
        print("accuracy = %0.8f for n_neighbors: %d" %(moy, i))
        
        clf_kf.fit(X,y)
        y_pred_kf = clf_kf.predict(Xtest)
        
        if (moy > best_score):
            best_i, best_score = i, moy
        clfs[i] = clf_kf
    
    best_clf = clfs[best_i]
    best_clf.fit(X, y)
    # determine the optimal parameters
    y_pred_best = best_clf.predict(Xtest)
    print("\nBest hyperparameter on the training set : n_neighbors = %d" %best_i)
    print("On the test set : %0.8f" \
        % metrics.accuracy_score(ytest, y_pred_best))