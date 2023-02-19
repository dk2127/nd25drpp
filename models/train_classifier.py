#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
For the machine learning portion, program splits the data into a training set and a
test set. Then, creates a machine learning pipeline that uses NLTK, as well as
scikit-learn's Pipeline and GridSearchCV to output a final model that uses the message
column to predict classifications for 30+ categories (multi-output classification).
Finally, exports the model to a pickle file.

Created on Feb 2023

@author: DK
"""

# In[ ]:


# import libraries
import sys
import re
import time
import pickle

import pandas as pd
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
#from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = stopwords.words("english")


# In[ ]:


def load_data(database_filepath):
    """
    Load data from a SQLite database file and split it into predictor (X) and response (y) variables.

    Args:
        database_filepath (str): Path to the SQLite database file.

    Returns:
        tuple: A tuple of three values:
               - A numpy array containing the predictor variables (messages)
               - A pandas dataframe containing the response variables (35 categories)
               - A numpy array containing the names of the response variable categories
    """

    # load data from database
    engine = create_engine("sqlite:///" + database_filepath)
    connection = engine.connect()
    query = "SELECT * FROM DisasterResponse"
    df = pd.read_sql(query, connection)
    connection.close()

    # Keep only the predictors in the X
    predictors = ["message"]
    X = df[predictors].message.values
    print("Dimensions of X are:", X.ndim)
    print("Shape of X is", X.shape)
    print("Size of X is", X.size)

    # keep ony the 35 response variables in y; dropped child_alone since all values are 0
    y = df.loc[:, ~df.columns.
               isin(['id', 'message', 'original', 'genre', 'child_alone'])]
    y.head()
    print("Dimensions of y are:", y.ndim)
    print("Shape of y is", y.shape)
    print("Size of y is", y.size)

    category_names = y.columns

    return X, y, category_names

# In[ ]:


def tokenize(text):
    """
    Tokenize text by removing punctuation, converting to lowercase, removing stop words, and lemmatizing.

    Args:
        text (str): The text to be tokenized.

    Returns:
        list: A list of clean tokens.
    """
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    tokens = word_tokenize(text)
    # remove stop words and lemmatize
    tokens = [w for w in tokens if w not in stopwords.words("english")]

    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# In[ ]:


def build_model():
    """
    Build a text classification model using a logistic regression classifier and a pipeline that applies
    CountVectorizer, TfidfTransformer, and MultiOutputClassifier to the text data.

    Returns:
        GridSearchCV: A GridSearchCV object that applies cross-validation and grid search to find the best
                      hyperparameters for the text classification model.
    """
    logreg = LogisticRegression()  # multi_class='ovr')

    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(logreg))])

    # from sklearn.model_selection import GridSearchCV
    print(pipeline.get_params().keys())

    parameters = {
        'clf__estimator__multi_class': ['ovr'],
        'clf__estimator__solver':
        ['saga'],  # Algorithm to use in the optimization problem
        'clf__estimator__n_jobs':
        [2],  # Number of CPU cores used when parallelizing over classes
        'clf__estimator__warm_start': [
            True
        ],  # When set to True, reuse the solution of the previous call to fit as initialization'
        'clf__estimator__penalty': ['l2', 'elasticnet']
        # ,'clf__estimator__verbose': [1] # commenting to reduce verbose o/p
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


# In[ ]:


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the performance of the trained model on the test set using classification report and confusion matrix
    for each response variable category.

    Args:
        model (object): Trained model object to be evaluated.
        X_test (numpy array): Numpy array containing the test set predictor variables.
        Y_test (pandas dataframe): Pandas dataframe containing the test set response variables (35 categories).
        category_names (numpy array): Numpy array containing the names of the response variable categories.

    Returns:
        None
    """

    y_pred = model.predict(X_test)

    # Classification Report for reponse variable
    i = 0
    for col in category_names:
        print(
            "\n--------------Classification Report for reponse variable#{}. {} ----------------"
            .format(i, col))
        print(classification_report(Y_test[col], y_pred[:, i]))
        i += 1

    # confusion_matrix for reponse variables

    #i = 0
    #for col in category_names:
    #    cm = confusion_matrix(Y_test[col],
    #                          y_pred[:, i],
    #                          labels=model.classes_[i])
    #    fig, ax = plt.subplots()
    #    ConfusionMatrixDisplay(cm,
    #                           display_labels=model.classes_[i]).plot(ax=ax)
    #    ax.set_title(f"Confusion Matrix for {col}")
    #    i += 1


# In[ ]:


def save_model(model, model_filepath):
    """
    Save a trained machine learning model to a pickle file.

    Args:
        model: A trained machine learning model.
        model_filepath (str): Path to the output pickle file.

    Returns:
        None
    """

    with open(model_filepath, 'wb') as pklfile:
        pickle.dump(model, pklfile)


# # ------------Begin comment to test standalone --------------------------------------------
# # Test e2e
# database_filepath ='../data/DisasterResponse.db'
# model_filepath = 'classifier.pkl'
#
# print('Loading data...\n    DATABASE: {}'.format(database_filepath))
# X, Y, category_names = load_data(database_filepath)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
#
# print('Building model...')
# model = build_model()
#
# print('Training model...')
# # start time
# starttm = time.time()
#
# # fit model
# model.fit(X_train, Y_train)
#
# # end time
# endtm =  time.time()
# execTmMilsec = (endtm-starttm) * 10**3
#
# print("execution time for the fit=",execTmMilsec,"Millseconds" )
#
# print('Evaluating model...')
# evaluate_model(model, X_test, Y_test, category_names)
#
# print('Saving model...\n    MODEL: {}'.format(model_filepath))
# save_model(model, model_filepath)
#
# print('Trained model saved!')
# # ------------End comment  to test standalone --------------------------------------------

# In[ ]:


def main():
    """
   The main function of the script that trains a machine learning model to classify messages related to
   natural disasters and saves the model as a pickle file.

   This function loads the data from a SQLite database file, splits it into training and testing datasets,
   builds a machine learning model, trains the model on the training data, evaluates the model on the testing
   data, saves the trained model as a pickle file, and prints the execution time for the model fitting.

   Args:
       None

   Returns:
       None
   """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                            Y,
                                                            test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        # start time
        starttm = time.time()

        # fit model
        model.fit(X_train, Y_train)

        # end time
        endtm = time.time()
        execTmMilsec = (endtm - starttm) * 10**3

        print("execution time for the fit=", execTmMilsec, "Millseconds")

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
