import logistic_regression as log_reg
import svm as svm
import data as d
import aux as a
import numpy as np
import random

df = d.read_file_columns()

train_df,valid_df,test_df = d.split_groups(df)

x_train_features,y_train_values = d.get_x_y(train_df)
x_valid_features,y_valid_values = d.get_x_y(valid_df)
x_test_features,y_test_values = d.get_x_y(test_df)

def algorithm_svm1(epochs,num_tests,k=7):
    alpha_nums = []
    const_nums = []
    for i in range(num_tests):
        """ alpha = random.uniform(0.01,0.05)
        alpha_nums.append(alpha) 
        constant = random.uniform(0.5,1.5)
        const_nums.append(constant) """
        alpha = 0.05
        constant = 1
        error_train_list,error_valid_list = svm.regression(
            x_train_features,y_train_values,x_valid_features,y_valid_values,alpha,epochs,constant,k)
        
        a.plot_error(epochs,error_train_list,error_valid_list,f"svm{i}")

    #a.save_files(alpha_nums,const_nums)

algorithm_svm1(100, 5)
