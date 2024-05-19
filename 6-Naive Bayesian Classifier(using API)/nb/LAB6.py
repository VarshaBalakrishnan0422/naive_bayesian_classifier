import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

# Title of the web app
st.title('Naive Bayes Text Classifier')

# File uploader to upload the CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    msg = pd.read_csv(uploaded_file, names=['message', 'label'])
    
    # Display the total instances of the dataset
    st.write("Total Instances of Dataset: ", msg.shape[0])
    
    # Map labels to numerical values
    msg['labelnum'] = msg.label.map({'pos': 1, 'neg': 0})
    
    # Features and labels
    X = msg.message
    y = msg.labelnum
    
    # Split data into training and test sets
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Vectorize text data
    count_v = CountVectorizer()
    Xtrain_dm = count_v.fit_transform(Xtrain)
    Xtest_dm = count_v.transform(Xtest)
    
    # Create DataFrame from training data matrix
    df = pd.DataFrame(Xtrain_dm.toarray(), columns=count_v.get_feature_names_out())
    st.write("Sample of Training DataFrame: ", df.head())
    
    # Train Naive Bayes classifier
    clf = MultinomialNB()
    clf.fit(Xtrain_dm, ytrain)
    
    # Predict on test data
    pred = clf.predict(Xtest_dm)
    
    # Print predictions for test data
    st.write("Predictions on Test Data:")
    results = []
    for doc, p in zip(Xtest, pred):
        p_label = 'pos' if p == 1 else 'neg'
        results.append(f"{doc} -> {p_label}")
    st.write("\n".join(results))
    
    # Calculate and print accuracy metrics
    st.write('Accuracy Metrics: \n')
    st.write('Accuracy: ', accuracy_score(ytest, pred))
    st.write('Recall: ', recall_score(ytest, pred))
    st.write('Precision: ', precision_score(ytest, pred))
    st.write('Confusion Matrix: \n', confusion_matrix(ytest, pred))
