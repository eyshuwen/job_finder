#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st

# Sample data for illustration
data = pd.read_csv(r'C:\Users\Eyshuwen\Google Drive\SHUWEN~1\Jobs\PYTHON~1\PREASS~1\ASSESS~1\ASSESS~1\SECTIO~1.CSV')
ssoc_data = pd.DataFrame(data)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(ssoc_data['Job Title'], ssoc_data['Labelled SSOC Title'], test_size=0.3, random_state=42)

# Convert text data into TF-IDF features
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Predict the SSOC codes for the test set
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Display evaluation results in the console
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)

# Function to predict SSOC code for a given job title
def predict_ssoc(job_title):
    job_title_tfidf = vectorizer.transform([job_title])
    ssoc_code = model.predict(job_title_tfidf)[0]
    return ssoc_code

# Streamlit User Interface
st.title("SSOC Code Finder")

# Text input for job title
job_title = st.text_input("Enter Job Title:")

# Button to find SSOC code
if st.button("Find SSOC Code"):
    ssoc_code = predict_ssoc(job_title)
    st.write(f"The SSOC code for '{job_title}' is: {ssoc_code}")


# In[ ]:


get_ipython().system('streamlit run C:\\Users\\Eyshuwen\\anaconda3\\lib\\site-packages\\ipykernel_launcher.py')


# In[ ]:




