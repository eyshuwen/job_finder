#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd

# Load the dataset
file_path = r'C:\Users\Eyshuwen\Google Drive\Shu Wen\Test\Test.csv'
df = pd.read_csv(file_path)

# Function to find SSOC classification based on job title
def find_ssoc_classification(job_title):
    matching_row = df[df['Job Title'].str.contains(job_title, case=False, na=False)]
    if not matching_row.empty:
        ssoc_code = matching_row.iloc[0]['Labelled SSOC']
        ssoc_title = matching_row.iloc[0]['Labelled SSOC Title']
        return ssoc_code, ssoc_title
    else:
        return None, None

# Function to find similar job listings
def find_similar_jobs(job_title):
    similar_jobs = df[df['Job Title'].str.contains(job_title, case=False, na=False)]
    return similar_jobs

# Streamlit app
st.title('Job Title to SSOC Classification Finder')

# Input job title
job_title_input = st.text_input('Enter Job Title:')

if job_title_input:
    ssoc_code, ssoc_title = find_ssoc_classification(job_title_input)
    
    if ssoc_code:
        st.write(f"SSOC Classification: {ssoc_code} - {ssoc_title}")
    else:
        st.write("No SSOC classification found. You can skip this field.")
    
    st.subheader('Potential Job Listings:')
    similar_jobs = find_similar_jobs(job_title_input)
    if not similar_jobs.empty:
        for index, row in similar_jobs.iterrows():
            st.write(f"Job Title: {row['Job Title']}")
            st.write(f"SSOC Classification: {row['Labelled SSOC']} - {row['Labelled SSOC Title']}")
            st.write('---')
    else:
        st.write("No similar job listings found.")

