import logistic_regression as log
import svm as svm
import knn as knn
import data as d
import aux as a
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
import subprocess
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score

#################
###### SVM ######
#################

df_svm = d.read_file_columns("svm")

train_df_svm,valid_df_svm,test_df_svm = d.split_groups(df_svm)

x_train_features_svm,y_train_values_svm = d.get_x_y(train_df_svm)
x_valid_features_svm,y_valid_values_svm = d.get_x_y(valid_df_svm)
x_test_features_svm,y_test_values_svm = d.get_x_y(test_df_svm)

def algorithm_svm1(epochs,num_tests,k=7):
    for i in range(num_tests):
        alpha = 0.05
        constant = 1
        error_train_list,error_valid_list = svm.regression(
            x_train_features_svm,y_train_values_svm,
            x_valid_features_svm,y_valid_values_svm,alpha,epochs,constant,k)
        
        a.plot_error(epochs,error_train_list,error_valid_list,f"svm{i}")
        
def algorithm_svm2(epochs,num_tests,k=7):
    for i in range(num_tests):
        alpha = 0.05
        constant = 1
        error_train_list,error_valid_list,error_test_list = svm.regression2(
            x_train_features_svm,y_train_values_svm,
            x_valid_features_svm,y_valid_values_svm,
            x_test_features_svm,y_test_values_svm,alpha,epochs,constant,k)
        
        a.plot_error2(epochs,error_train_list,error_valid_list,error_test_list,f"svm_test{i}")

""" algorithm_svm1(300, 2)
algorithm_svm2(300, 2) """

#############################
#### LOGISTIC REGRESSION ####
#############################

df_log = d.read_file_columns("logistic")

train_df_log,valid_df_log,test_df_log = d.split_groups(df_log)

x_train_features_log,y_train_values_log = d.get_x_y(train_df_log)
x_valid_features_log,y_valid_values_log = d.get_x_y(valid_df_log)
x_test_features_log,y_test_values_log = d.get_x_y(test_df_log)

def algorithm_logistic1(epochs,num_tests,k=7):
    for i in range(num_tests):
        alpha = 0.05
        error_train_list,error_valid_list = log.regression(
            x_train_features_log,y_train_values_log,
            x_valid_features_log,y_valid_values_log,alpha,epochs,k)
        
        a.plot_error(epochs,error_train_list,error_valid_list,f"logistic{i}")

def algorithm_logistic2(epochs,num_tests,k=7):
    for i in range(num_tests):
        alpha = 0.05
        error_train_list,error_valid_list,error_test_list = log.regression2(
            x_train_features_log,y_train_values_log,
            x_valid_features_log,y_valid_values_log,
            x_test_features_log,y_test_values_log,alpha,epochs,k)
        
        a.plot_error2(epochs,error_train_list,error_valid_list,error_test_list,f"logistic_test{i}")

""" algorithm_logistic1(300, 2)
algorithm_logistic2(300, 2) """

#############################
############ KNN ############
#############################

def algorithm_knn(neighbors):
    experiments = 10
    errors_list = []
    confusion_list = []
    for i in range(experiments):
        df_exp, df_test = d.cross_validation(df_log, experiments, i)
        number_errors, temp_mat = knn.knn(df_exp.to_numpy()[:,1:], df_test.to_numpy()[:,1:], neighbors)
        errors_list.append(number_errors)
        confusion_list.append(temp_mat)
    accuracy_score = confusion_list[0]
    print("     F    M")
    print("F  ", accuracy_score[0])
    print("M  ", accuracy_score[1])
    accuracy = (accuracy_score[0][0] + accuracy_score[1][1]) / (sum(accuracy_score[0]) + sum(accuracy_score[1]))
    print("Accuracy: ", accuracy)
    #a.plot_error_knn(errors_list)

algorithm_knn(5)


############################
###### DECISION TREES ######
############################

df_tree = d.read_file_columns("decision_tree")

train_df_tree,valid_df_tree,test_df_tree = d.split_groups(df_tree)

x_train_features_tree,y_train_values_tree = d.get_x_y(train_df_tree)
x_valid_features_tree,y_valid_values_tree = d.get_x_y(valid_df_tree)
x_test_features_tree,y_test_values_tree = d.get_x_y(test_df_tree)

def build_tree():
    class_names = ["Male","Female"]
    des_tree = tree.DecisionTreeClassifier(min_samples_split=50,splitter='best')
    model = des_tree.fit(x_train_features_tree, y_train_values_tree)
    y_predict = model.predict(x_valid_features_tree)
    accuracy = accuracy_score(y_valid_values_tree, y_predict)
    print("Classification report:")
    print(classification_report(y_valid_values_tree, y_predict, target_names=class_names))
    print("Accuracy:")
    print(accuracy)
    print(" ")
    print("Confusion Matrix:")
    print(confusion_matrix(y_valid_values_tree, y_predict))

    with open("dt.dot", 'w') as f:
            tree.export_graphviz(des_tree, out_file=f,feature_names = list(x_train_features_tree.columns))
    command = ["dot", "-Tpng", "plots/dt.dot", "-o", "plots/dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
                "produce visualization")

#build_tree()
