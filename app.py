import numpy as np 
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv('Job_Placement_Data.csv')
df_clean = df.copy()
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,8))   # This is a filter method useful to select features. With the help oh heatmap we will take only the subset of relevant feature means the feature that are relevent 
cor = df_clean.corr()      # we will select only those that has correlation more than 0.5 because closer value to 1 shows more positive coorelation
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

ssc_board_type = pd.CategoricalDtype(categories = ['Others', 'Central'])
df_clean['ssc_board'] = df_clean['ssc_board'].astype(ssc_board_type)
gender_type = pd.CategoricalDtype(categories = ['M', 'F'])
df_clean['gender'] = df_clean['gender'].astype(gender_type)
hsc_board_type = pd.CategoricalDtype(categories = ['Others', 'Central'])
df_clean['hsc_board'] = df_clean['hsc_board'].astype(hsc_board_type)
undergrad_degree_type = pd.CategoricalDtype(categories = ['Sci&Tech', 'Comm&Mgmt', 'Others'])
df_clean['undergrad_degree'] = df_clean['undergrad_degree'].astype(undergrad_degree_type)
work_experience_type = pd.CategoricalDtype(categories = ['No', 'Yes'])
df_clean['work_experience'] = df_clean['work_experience'].astype(work_experience_type)
specialisation_type = pd.CategoricalDtype(categories = ['Mkt&HR', 'Mkt&Fin'])
df_clean['specialisation'] = df_clean['specialisation'].astype(specialisation_type)
status_type = pd.CategoricalDtype(categories = ['Placed', 'Not Placed'])
df_clean['status'] = df_clean['status'].astype(status_type)
# Replacing 'M' with'Male' and 'F' with 'Female'
df_clean['gender'].replace('M','Male' ,inplace = True)
df_clean['gender'].replace('F','Female' ,inplace = True)
df_clean = df_clean.drop(columns=['mba_percent','hsc_subject','hsc_board'])
df_clean['gender'].value_counts()
df_clean['status'].value_counts()


from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
df_clean['gender'] = lab.fit_transform(df_clean['gender']) 
df_clean['work_experience'] = lab.fit_transform(df_clean['work_experience'])
df_clean['specialisation'] = lab.fit_transform(df_clean['specialisation'])
#df_clean['status'] = lab.fit_transform(df_clean['status'])
df_clean['ssc_board'] = lab.fit_transform(df_clean['ssc_board'])
df_clean['undergrad_degree'] = lab.fit_transform(df_clean['undergrad_degree'])
df_clean.head(5)


df_clean = df_clean.rename(columns={'gender':'Gender','status':'Enrol_Ind','ssc_percentage':'Age','undergrad_degree':'Country','degree_percentage':'Matric_Ind','ssc_board':'Admit_Ind','work_experience':'Waitlisted_Ind','emp_test_percentage':'Cancelled_Ind','specialisation':'Academic_Group_Code','hsc_percentage':'Trending_Term'})

import streamlit as st
from sklearn.model_selection import cross_val_score, KFold
df_clean.describe()
y_names = ['Enrolled', 'Not enrolled']
df_clean['Enrol_Ind'] = df_clean['Enrol_Ind'].replace({'Enrolled': 0, 'Not Enrolled': 1})


import numpy as np
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

st.title("student enrollement Prediction")

st.sidebar.header('User Input Parameters')

def user_input_features():
  Gender = st.sidebar.slider('Gender', 0.0, 1.0, 0.6)
  Age = st.sidebar.slider('Age', 40.8,89.4, 67.3)
  Admit_Ind = st.sidebar.slider('Admit_Ind', 0.0, 1.0, 0.4)
  Trending_Term = st.sidebar.slider('Trending_Term', 37.0, 97.7, 66.3)
  Matric_Ind = st.sidebar.slider('Matric_Ind', 50.0,97.0, 66.3)
  Country = st.sidebar.slider('Country', 0.0,2.0 , 0.6)
  Waitlisted_Ind = st.sidebar.slider('Waitlisted_Ind', 0.0, 1.0, 0.3)
  Cancelled_Ind = st.sidebar.slider('Cancelled_Ind', 50.0, 98.0, 72.1)
  Academic_Group_Code = st.sidebar.slider('Academic_Group_Code', 0.0, 1.0, 0.4)

  user_input_data = {'Gender': Gender,
               'Age': Age,
               'Admit_Ind': Admit_Ind,
               'Trending_Term': Trending_Term,
               'Matric_Ind': Matric_Ind,
               'Country': Country,
               'Waitlisted_Ind': Waitlisted_Ind,
               'Cancelled_Ind':Cancelled_Ind,
               'Academic_Group_Code': Academic_Group_Code}
              
  features = pd.DataFrame(user_input_data, index=[0])
  return features

df = user_input_features()

st.subheader('User Input Parameters')
st.write(df)


X = df_clean.loc[:, ['Gender', 'Age', 'Admit_Ind', 'Trending_Term', 'Matric_Ind', 'Country',
       'Waitlisted_Ind', 'Cancelled_Ind', 'Academic_Group_Code']]

y = df_clean['Enrol_Ind']

# Encode the Target variable
le = LabelEncoder()
y = le.fit_transform(y)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

# Model Prediction
prediction = model.predict(df)

# Output
target_array = df_clean['Enrol_Ind'].values
#st.subheader('Prediction')
#st.write(target_array[prediction])

prediction_probabilities = model.predict_proba(df)
#st.subheader('Class labels and their corresponding index number')
#st.write(target_array)

st.subheader('Prediction Probability')
st.write(prediction_probabilities)



