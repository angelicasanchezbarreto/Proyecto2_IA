import logistic_regression as log
import svm as svm
import knn as knn
import data as d
import aux as a
import random
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

#################
###### SVM ######
#################

df = d.read_file_columns("svm")

train_df,valid_df,test_df = d.split_groups(df)

x_train_features,y_train_values = d.get_x_y(train_df)
x_valid_features,y_valid_values = d.get_x_y(valid_df)
x_test_features,y_test_values = d.get_x_y(test_df)


def algorithm_svm1(epochs,num_tests,k=7):
    for i in range(num_tests):
        alpha = 0.05
        constant = 1
        error_train_list,error_valid_list,w = svm.regression(
            x_train_features,y_train_values,x_valid_features,y_valid_values,alpha,epochs,constant,k)
        
        a.plot_error(epochs,error_train_list,error_valid_list,f"svm{i}")

    #a.svm(x_train_features,y_train_values,w)
    
    #a.save_files(alpha_nums,const_nums)

#algorithm_svm1(500, 1)

#############################
#### LOGISTIC REGRESSION ####
#############################

df = d.read_file_columns("logistic")

train_df,valid_df,test_df = d.split_groups(df)

x_train_features,y_train_values = d.get_x_y(train_df)
x_valid_features,y_valid_values = d.get_x_y(valid_df)
x_test_features,y_test_values = d.get_x_y(test_df)

def algorithm_logistic1(epochs,num_tests,k=7):
    for i in range(num_tests):
        alpha = 0.05
        error_train_list,error_valid_list = log.regression(
            x_train_features,y_train_values,x_valid_features,y_valid_values,alpha,epochs,k)
        
        a.plot_error(epochs,error_train_list,error_valid_list,f"logistic{i}")

#algorithm_logistic1(500, 1)

#############################
############ KNN ############
#############################

def algorithm_knn(neighbors):
    experiments = 10
    errors_list = []
    for i in range(experiments):
        df_exp, df_test = d.cross_validation(df, experiments, i)
        number_errors = knn.knn(df_exp.to_numpy()[:,1:], df_test.to_numpy()[:,1:], neighbors)
        errors_list.append(number_errors)
    return errors_list

errors = algorithm_knn(5)
print(np.var(errors))
print(np.mean(errors))

sns.kdeplot(errors, shade = True)
plt.show()

""" sns.distplot(x = errors, rug=True,
             axlabel="Something ?",
             kde_kws=dict(label="kde"),
             rug_kws=dict(height=.2, linewidth=2, color="C1", label="data"))
plt.legend(); """
