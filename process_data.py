import sys
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,on='id')
    return df, categories

def clean_data(df,categories):
    # create a dataframe of the 36 individual category columns
    categories = categories.categories.str.split(';',expand=True)
    col_list=list(categories.iloc[0,:])
    i=0
    for col in col_list:
        col_list[i]=col.split('-')[0]
        i=i+1
    categories.set_axis(col_list,axis=1,inplace=True)
    #Replace categories column in df with new category columns and convert to 0 and 1
    categories=categories.applymap(lambda x: int(str(x).split('-')[1]))
    categories[categories>1]=1
    # drop the original categories column from `df`
    df.drop(columns='categories',inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df[categories.columns]=categories
    # drop duplicates
    df.drop_duplicates(inplace=True)

def save_data(df, database_filename):
    engine = create_engine('sqlite:///{}.db'.format(database_filename))
    df.to_sql(database_filename, engine, index=False,if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df,categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df,categories)
        
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