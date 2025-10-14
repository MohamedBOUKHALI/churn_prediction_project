import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score
from sklearn.preprocessing import MinMaxScaler


def Load_Mydata(file):
    data = pd.read_csv(file)
    return data
#print(Load_Mydata('data-68e11476082f9096032105.csv').head())

def Check_Data(data):
    print(data.head())
    print(data.info())
    print(data.describe())
    print(data.isnull().sum())
    print(data.duplicated().sum())
#Check_Data(Load_Mydata('data-68e11476082f9096032105.csv'))

def Numerical_Column_TotalChargers(data):
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    return data
#print(Numerical_Column_TotalChargers(Load_Mydata('data-68e11476082f9096032105.csv')).info())

def Fix_Missing_Values(data):
    data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].mean())
    return data

def Split_Target(data, target_column):
    My_data_num = Numerical_Column_TotalChargers(data)
    X = My_data_num.drop(columns=[target_column])
    y = My_data_num[target_column]
    return X, y
#print(Split_Target(Load_Mydata('data-68e11476082f9096032105.csv'), 'Churn'))

def Drop_non_important_Columns(data, columns_to_drop):
    data = data.drop(columns=columns_to_drop)
    return data
#print(Drop_non_important_Columns(Load_Mydata('data-68e11476082f9096032105.csv'), ['customerID','Churn','gender','tenure']).head())

def Normalized_Columns(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    return data
#print(Normalized_Columns(Load_Mydata('data-68e11476082f9096032105.csv')).head())

def Encode_Categorical_Columns(data):
    label_encoder = LabelEncoder()
    X_befor_Drop,Y = Split_Target(data, 'Churn')
    X_befor_Normalized = Drop_non_important_Columns( X_befor_Drop,['customerID','gender','tenure'])
    X_add_value = Fix_Missing_Values(X_befor_Normalized)
    X = Normalized_Columns(X_add_value)
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = label_encoder.fit_transform(X[col])
    return X,Y
#print(Encode_Categorical_Columns(Load_Mydata('data-68e11476082f9096032105.csv')))

def Split_Data(data):
    X,Y = Encode_Categorical_Columns(data)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
#print(Split_Data(Load_Mydata('data-68e11476082f9096032105.csv')))

Models = ['RandomForest', 'LogisticRegression']
def Train_Model(data,Models):
    Data_of_Models = []
    X_train, X_test, y_train, y_test = Split_Data(data)
    for model in Models:
        if model == 'RandomForest':
            my_model = RandomForestClassifier(random_state=42)
        elif model == 'LogisticRegression':
            my_model = LogisticRegression(max_iter=1000,random_state=42)
        my_model.fit(X_train, y_train)
        y_pred = my_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, pos_label='Yes')
        Data_of_Models.append({
            'Model': model,
            'Accuracy': accuracy,
            'F1_Score': f1
        })
    return pd.DataFrame(Data_of_Models)
#print(Train_Model(Load_Mydata('data-68e11476082f9096032105.csv'),Models))



def Final_result(data):
    results = Train_Model(data, Models)
    results.plot(
        x='Model',
        y=['F1_Score', 'Accuracy'],
        kind='bar',
        title='Model Performance Comparison',
        rot=0,
    )

    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()
# Final_result(Load_Mydata('data-68e11476082f9096032105.csv'))


data = Load_Mydata('data-68e11476082f9096032105.csv')
Final_result(data)







