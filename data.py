from os import X_OK
import pandas as pd
import math

def normalize_data(data):
    # Assuming same lines from your example
    cols_to_norm = ['forehead_width_cm','forehead_height_cm']
    data[cols_to_norm] = data[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    return data

def update_gender(df,method):
    if method=="svm":
        df.loc[df["gender"] == 'Male', "gender"] = -1
        df.loc[df["gender"] == 'Female', "gender"] = 1
    elif method=="logistic":
        df.loc[df["gender"] == 'Male', "gender"] = 0
        df.loc[df["gender"] == 'Female', "gender"] = 1
    else:
        df.loc[df["gender"] == 'Male', "gender"] = 1
        df.loc[df["gender"] == 'Female', "gender"] = 2


def plot_data(data_x,data_y):
    pd.plt.plot(data_x,data_y,'*')
    pd.plt.show()

def normalize_features(x_features):
    for column in x_features.columns[1:]:
        low_value = x_features[column].max()-x_features[column].min() if x_features[column].max()-x_features[column].min() != 0 else 0
        if low_value == 0:
            return 0
        else:
            return (x_features[column]-x_features[column].min())/low_value

##################
# MAIN FUNCTIONS #
##################

def read_file_columns(method,file = 'gender_classification.csv'):
    data = pd.read_csv(file)
    update_gender(data,method)
    data = normalize_data(data)
    return data

def get_x_y(df):
    x_features = df.iloc[:,:-1]
    y_values = df.iloc[:,-1]
    return x_features,y_values

def split_groups(df):
    df = df.sample(frac = 1).reset_index(drop = True)
    train_df = pd.DataFrame(columns=df.columns)
    valid_df = pd.DataFrame(columns=df.columns)
    test_df = pd.DataFrame(columns=df.columns)
    
    a,b,c = 0,0,0
    for index, row in df.iterrows():
        if index <= math.ceil(df.shape[0] * 0.7):
            train_df.loc[a] = row.values
            a += 1
        elif index > math.ceil(df.shape[0] * 0.7) and index <= math.floor(df.shape[0] * 0.9):
            valid_df.loc[b,:] = row.values
            b += 1
        else:
            test_df.loc[c,:] = row.values
            c += 1
            
    return train_df,valid_df,test_df

def cross_validation(df, exp, num_exp):
    df = df.sample(frac = 1).reset_index(drop = True)
    df_exp = []
    cont = 0
    df_size = math.floor(len(df) / 10)
    for i in range(exp):
        if i == (exp - 1):
            df_temp = df.iloc[cont : len(df)]
            df_exp.append(df_temp.reset_index())
        else:   
            df_temp = df.iloc[cont : (cont + df_size)]
            cont += df_size
            df_exp.append(df_temp.reset_index())
    df_test = df_exp.pop(num_exp)
    df_exp = pd.concat(df_exp, ignore_index = True)
    return df_exp, df_test
