#!/usr/bin/env python3


import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools
import kernel
import itertools
from kernel import Kernel
import plotting
from plotting import Plotting
import generate_data
from generate_data import Data
import svm
from svm import SVM



def example(num_samples=100, num_features=2, grid_size=200, filename="svm.pdf"):
    Data_obj = Data();
    Plot_obj = Plotting();    
    
    X_1, y_1, X_2, y_2 = Data_obj.gen_non_lin_separable_data(seed=1)
    clf = SVM(kernel=Kernel.rbf(0.1),kernel_type = 'gaussian', C=100, gamma=0.001, degree=5)  
    #Split dataset into training and testing samples                                           
    X_train, y_train, X_test, y_test = Data.split_data(X_1, y_1, X_2, y_2,0.8) 
    clf.fit(X_train, y_train)
    #plot(clf, X_train, y_train, grid_size, "svm3.pdf")
    Plot_obj.plot_margin(X_train, y_train, clf);

    X_1, y_1, X_2, y_2 = Data_obj.gen_lin_separable_overlap_data(seed=3)
    clf = SVM(kernel=Kernel.rbf(0.1),kernel_type = 'polynomial', C=100, gamma=0.001, degree=3)  
    #Split dataset into training and testing samples                                           
    X_train, y_train, X_test, y_test = Data.split_data(X_1, y_1, X_2, y_2,0.8) 
    clf.fit(X_train, y_train)
    #plot(clf, X_train, y_train, grid_size, "svm3.pdf")
    Plot_obj.plot_margin(X_train, y_train, clf);
 




def plot(predictor, X, y, grid_size, filename):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                         np.linspace(y_min, y_max, grid_size),
                         indexing='ij')
    flatten = lambda m: np.array(m).reshape(-1,)

    result = []
    for (i, j) in itertools.product(range(grid_size), range(grid_size)):
        point = np.array([xx[i, j], yy[i, j]]).reshape(1, 2)
        result.append(predictor.predict(point))

    Z = np.array(result).reshape(xx.shape)
    plt.clf()
    plt.contourf(xx, yy, Z,
                 cmap=cm.Accent,
                 levels=[-0.0001, 0.0001],
                 extend='both',
                 alpha=0.4)
    plt.scatter(flatten(X[:, 0]), flatten(X[:, 1]),
                c=flatten(y), cmap=cm.Paired)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.savefig(filename)


if __name__ == "__main__":
    example()
  