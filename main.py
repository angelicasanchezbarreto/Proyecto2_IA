import logistic_regression as log
import svm as svm
import knn as knn
import data as d
import aux as a
import random

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

algorithm_logistic1(500, 1)

#############################
############ KNN ############
#############################

df = d.read_file_columns("logistic")

train_df,valid_df,test_df = d.split_groups(df)

train_df = train_df.to_numpy()
valid_df = valid_df.to_numpy()
test_df = test_df.to_numpy()

def algorithm_knn(num_tests):
    for i in range(num_tests):
        number_errors = knn.KNN(train_df, valid_df, 4)
    return number_errors