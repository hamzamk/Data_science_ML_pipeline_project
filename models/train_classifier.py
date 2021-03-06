'''
This file requires sklearn 0.23.1 and the correct version will be automatically installed if the necessary method can't be imported. The best parameters for the model are selected as a result of a grid search hence the search algorithm will not be rerun. 

'''


import os
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import re
import sys
import string
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import seaborn as sns
from numpy import argmax
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import pickle

# multilabel_confusion_matrix is not present in scikit-learn version < 0.23.1
try:
    from sklearn.metrics import multilabel_confusion_matrix
except ImportError:
    os.system('pip install -U scikit-learn')
finally:
    from sklearn.metrics import multilabel_confusion_matrix

def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    categories = df.iloc[:, 4:].columns
    X, Y = df.message, df.iloc[:,4:]
    return X, Y, categories


def tokenize(text):
    '''
    Cleans URLs, punctuation, normalized text, creates tokens and lemmatize text
    '''
    url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(url_regex, "", text).lower().split()
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in text]
    tokens = word_tokenize(' '.join(stripped))
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok)
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('svd', TruncatedSVD()),
    ('clf', MultiOutputClassifier(RandomForestClassifier())),
])
    # best parameters estimated using gridsearch
    parameters = {
    #    'tfidf__use_idf': (True, False), 
    #     'clf__estimator__n_estimators': [50, 100],
    #     'clf__estimator__min_samples_split': [2, 3, 4],
    'vect__max_df': [0.5],
    'vect__max_features': [5000],
    'vect__ngram_range': [(1, 2)],
    }
    grid = GridSearchCV(pipeline, cv=2, param_grid=parameters)
    return grid

def print_confusion_matrix(confusion_matrix, axes, class_label, class_names, fontsize=14):

    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    axes.set_xlabel('True label')
    axes.set_ylabel('Predicted label')
    axes.set_title("Class - " + class_label)


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    The classification report only shows two classes as 1 and 0 which are indicating boolean True and False for the labels. The metrics showcase how predicted labels match with the true labels. The summary of each class against another is depicted inthe confusion matrix which can be ploted or viewed in the terminal/command prompt
     '''
    y_pred = model.predict(X_test)
    y_test = np.array(Y_test)

    x = classification_report(np.concatenate(np.array(Y_test)), np.concatenate(y_pred))
    print('Classification report: ',x)

    confusion_mat = multilabel_confusion_matrix(y_test, y_pred)
    for matrix in tuple(zip( category_names, confusion_mat)):
        print(matrix)

    x = input('visualize confusion matrix? if there is no GUI then skip, else type "yes" ')
    
    if x == 'yes':
        fig, ax = plt.subplots(7, 5, figsize=(20, 20))    
        for axes, cfs_matrix, label in zip(ax.flatten(), confusion_mat, category_names):
            print_confusion_matrix(cfs_matrix, axes, label, ["Y", "N"])
        fig.tight_layout()
        plt.show()
    

def save_model(model, model_filepath):
    # Open the file to save as pkl file
    classifier = open(model_filepath, 'wb')
    pickle.dump(model, classifier)
    # Close the pickle instances
    classifier.close()


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
