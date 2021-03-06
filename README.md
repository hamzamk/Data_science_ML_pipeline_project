# Project description

This project utilized data provided by 'Figure Eight' which consists of messages generated as a result of a disaster. These messages consist of 36 categories such as medical help, weather, refugees, child-alone etc. The goal of this project is to create a multi-class multi-label classifier, integrated into a Flask web application. The data is cleaned by creating an ETL pipeline and then by using a Random Forest Classifier a multi-label output is generated for a given message.

# Installation Requirements
Python 3.6, sklearn 0.23.1, pandas, numpy, pickle, sqlalchemy, flask, re, string, nltk, seaborn, joblib


# how to run
- To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
- To run ML pipeline that trains classifier and saves the model
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/