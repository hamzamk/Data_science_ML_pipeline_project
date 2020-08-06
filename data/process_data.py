# import libraries
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine
import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    print(messages_filepath, categories_filepath)
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(categories, messages, how='outer', on=['id'])
    return df

def clean_data(df):
    '''
    There are rows which have all false values and do not add any meaningful information to the model and hence are removed. Similarly Child-alone class has no true labels and is recommended to be removed but kept.
    '''
    categories = df['categories'].str.split(';', expand = True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[0:-2])
    categories.columns = category_colnames
    for column in categories:
    # set each value to be the last character of the string
        categories[column] =  categories[column].str[-1]
    # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    df = df.drop(['categories', 'Unnamed: 0_y', 'Unnamed: 0_x'], axis=1)
    df =  pd.concat([df, categories], axis=1)
    df.duplicated(subset=None, keep='first').sum()
    df = df.drop_duplicates()
    for category in categories:
        if df[category].max() > 1:
            print('labels > 1 found in category: "', category, '" rows of that index shall be dropped')
            index_ = df.loc[df[category] > 1].index
            df.drop(index_, inplace = True)
    for category in categories:
        if df[category].unique().sum() < 1:
            print('column of class ' + category + 'has no "True" or 1 label' + ' this column can be dropped')
            #df.drop([category], axis = 1, inplace=True)
    return df

def save_data(df, database_filename):
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse', engine, index=False) 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

if __name__ == '__main__':
    main()
