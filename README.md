# nd25drpp
# Disaster Response Pipeline Project

Project Step
+ ETL, process_data.py : The first part of the data pipeline is the Extract, Transform, and Load process. Here, we will read the dataset, clean the data, and then store 
it in a SQLite database. We do the data cleaning with pandas. To load the data into an SQLite database, we use the pandas dataframe .to_sql() method, with an SQLAlchemy
engine.

  There is Jupyter notebook that has some exploratory data analysis to clean the data set. 

+ Machine Learning Pipeline, train_classifier.py : For the machine learning portion, we will split the data into a training set and a test set. Then, we will create 
a machine learning pipeline that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model that uses the message column to predict 
classifications for 36 categories (multi-output classification). Finally, we will export your model to a pickle file. 


+ Flask App, in the app folder has the starter files to display the results. The web app will need to upload the database file and pkl file with the model.

  Each file has the libraries imported as required.


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
