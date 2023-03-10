# Disaster Response Pipeline Project

## Introduction

In this project, we have a data set containing real messages that were sent during disaster events. We will be creating a machine learning pipeline to categorize these events so that we can send the messages to an appropriate disaster relief agency.

Our project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display a couple visualizations of the data.

This project showcases our software skills, including our ability to create basic data pipelines and write clean, organized code!


## Project Steps
+ ETL, process_data.py 
  * The first part of the data pipeline is the Extract, Transform, and Load process. Here, we will read the dataset, clean the data, and then store 
it in a SQLite database. We do the data cleaning with pandas. To load the data into an SQLite database, we use the pandas dataframe .to_sql() method, with an SQLAlchemy engine.

  * There are Jupyter notebooks that have exploratory data analysis for ETL.

+ Machine Learning Pipeline, train_classifier.py 
  * For the machine learning portion, we will split the data into a training set and a test set. Then, we will create a machine learning pipeline that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model that uses the message column to predict classifications for 30+ categories (multi-output classification). Finally, we will export your model to a pickle file.

  * Two Jupyter notebooks are included that were used for the classification model development using, Logistics Regression and SVM.


+ Flask App, run.py
  * The app folder has the starter files to display the results. The web app uses the database and pkl model files created in first two steps.

  * Each file has the libraries imported as required.


### Instructions:

### Please create directories with files in them, as they appear in this repo.

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
    ```
       python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
    ```    
    - To run ML pipeline that trains classifier and saves
    ```  
       python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
    ```
2. Run the following command in the app's directory to run your web app.
    ```
       python run.py
    ```
3. Go to http://0.0.0.0:3001/ on the workspace IDE

### nd25drpp
