# import streamlit as st
# import pandas as pd
# import seaborn as sns
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# header=st.container()
# data_sets=st.container()
# features=st.container()
# model_training=st.container()
# with header:
#     st.title('Machine Learning Web App with Streamlit')
#     st.text('Build with Streamlit and Python')
# with data_sets:
#     st.header('Data Sets')
#     st.text('Titanic Data Set')
#     df=sns.load_dataset('titanic')
#     df=df.dropna()
#     st.write(df.head())
#     st.subheader('According to Males and Females')
#     st.bar_chart(df['sex'].value_counts())
#     st.subheader('According to Class')
#     st.bar_chart(df['class'].value_counts()) 
#     st.bar_chart(df['age'].sample(5))
# with features:
#     st.header('Features')
#     st.text('Pclass - Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)')
#     st.markdown('**Survived** - Survival (0 = No; 1 = Yes)')
# with model_training:
#     st.header('Model Training')
#     st.text('Random Forest Classifier')
#     st.text('Accuracy: 0.82')
#     input,display=st.columns(2)
#     max_depth=input.slider('Max Depth',min_value=10,max_value=100,value=20,step=5)
# n_estimators=input.selectbox('N Estimators',options=[50,100,200,300,'None'])
# input.write(df.columns)
# input_features=input.text_input('Input Features')
# model=RandomForestRegressor(max_depth=max_depth,n_estimators=n_estimators)
# if n_estimators=='None':
#     model=RandomForestRegressor(max_depth=max_depth)
# else:
#     model=RandomForestRegressor(max_depth=max_depth,n_estimators=n_estimators)
# x=df[[input_features]]
# y=df[['fare']]
# model.fit(x,y)
# pred=model.predict(y)
# display.subheader('Mean Squared Error',mean_squared_error(y,pred))
# display.write('**Mean Squared Error**',mean_squared_error(y,pred))
# display.subheader('Mean Absolute Error',mean_absolute_error(y,pred))
# display.write('**Mean Absolute Error**',mean_absolute_error(y,pred))
# display.subheader('R2 Score',r2_score(y,pred))
# display.write('**R2 Score**',r2_score(y,pred))
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
st.write("""
# Simple Iris Flower Prediction App
This app predicts the **Iris flower** type!
""")
dataset_name=st.sidebar.selectbox('Select Dataset',('Iris','Breast Cancer','Wine'))
def get_dataset(dataset_name):
    data=None
    if dataset_name=='Iris':
        data=datasets.load_iris()
    elif dataset_name=='Breast Cancer':
        data=datasets.load_breast_cancer()
    else:
        data=datasets.load_wine()
    X=data.data
    y=data.target
    return X,y
X,y=get_dataset(dataset_name)
st.write('Shape of dataset',X.shape)
st.write('Number of classes',len(np.unique(y)))
def add_parameter_ui(clf_name):
    params=dict()
    if clf_name=='SVM':
        C=st.sidebar.slider('C',0.01,10.0)
        params['C']=C
    else:
        max_depth=st.sidebar.slider('max_depth',2,15)
        n_estimators=st.sidebar.slider('n_estimators',1,100)
        params['max_depth']=max_depth
        params['n_estimators']=n_estimators
    return params
params=add_parameter_ui(dataset_name)
def get_classifier(clf_name,params):
    clf=None
    if clf_name=='SVM':
        clf=SVC(C=params['C'])
    else:
        clf=RandomForestClassifier(n_estimators=params['n_estimators'],max_depth=params['max_depth'],random_state=1234)
    return clf
clf=get_classifier(dataset_name,params)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1234)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
acc=accuracy_score(y_test,y_pred)
st.write(f'Classifier={dataset_name}')
st.write(f'Accuracy={acc}')
pca=PCA(2)
X_projected=pca.fit_transform(X)
x1=X_projected[:,0]
x2=X_projected[:,1]
fig=plt.figure()
plt.scatter(x1,x2,c=y,alpha=0.8,cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()
st.pyplot(fig)
