# streamlit_app.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Function to load and process data
@st.cache_data
def load_data():
    data = pd.read_csv('document.csv')
    # Ensure the CSV has the correct headers
    if data.columns[0] != 'text' or data.columns[1] != 'label':
        data.columns = ['text', 'label']
    return data

# Function to train and evaluate the model
def train_and_evaluate(data):
    # Print the columns of the DataFrame to debug the issue
    st.write("Columns in the dataset:", data.columns)

    # Ensure the column names are correct
    if 'text' not in data.columns or 'label' not in data.columns:
        st.error("The dataset must contain 'text' and 'label' columns.")
        return None, None, None

    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train_vec, y_train)
    y_pred = nb_classifier.predict(X_test_vec)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    return accuracy, precision, recall

# Streamlit app
st.title('Naïve Bayes Document Classifier')

# Load data
data = load_data()

# Display data
if st.checkbox('Show raw data'):
    st.write(data)

# Train and evaluate model
if st.button('Train and Evaluate Model'):
    accuracy, precision, recall = train_and_evaluate(data)
    
    if accuracy is not None and precision is not None and recall is not None:
        st.write(f'**Accuracy:** {accuracy:.2f}')
        st.write(f'**Precision:** {precision:.2f}')
        st.write(f'**Recall:** {recall:.2f}')
