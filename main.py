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
    Plot_obj.plot_margin(X_train, y_train, clf);

    X_1, y_1, X_2, y_2 = Data_obj.gen_lin_separable_overlap_data(seed=3)
    clf = SVM(kernel=Kernel.rbf(0.1),kernel_type = 'polynomial', C=100, gamma=0.001, degree=3)  
    #Split dataset into training and testing samples                                           
    X_train, y_train, X_test, y_test = Data.split_data(X_1, y_1, X_2, y_2,0.8) 
    clf.fit(X_train, y_train)
    Plot_obj.plot_margin(X_train, y_train, clf);
 

if __name__ == "__main__":
    example()
  
