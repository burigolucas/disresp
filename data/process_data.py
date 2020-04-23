import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    INPUT:
    messages_filepath    - (str) filepath of the messages datasets
    categories_filepath  - (str) filepath of the categories datasets

    OUTPUT:
    df         - (pandas dataframe) df with messages and categories merged
    '''
    messages = pd.read_csv(messages_filepath).drop_duplicates(subset='id')
    categories = pd.read_csv(categories_filepath).drop_duplicates(subset='id')

    # merge datasets
    df = messages.merge(categories,on="id")

    return df


def clean_data(df):
    '''
    INPUT:
    df         - (pandas dataframe) df with messages and categories merged

    OUTPUT:
    df         - (pandas dataframe) clean df with categories split in separate columns
                                    duplicates are removed
    '''
    # create a dataframe of with all category columns
    categories = df['categories'].str.split(';',expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    category_colnames = [colname[:-2] for colname in row]
    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in category_colnames:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.slice(start=-1)

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

        if (categories[column].sum() == 0):
            print(f"Droping category {column} as it countains no entries")
            categories.drop(column,axis=1,inplace=True)

    # NOTE: column 'related' contains values which are within 0,1,2 causing issue to multilabel classification
    #       either convert 2 to 1 to resolve the issue, or remove the column from categories to be classified
    # categories.loc[categories['related'] == 2,'related'] = 1
    categories.drop('related',axis=1,inplace=True)

    # drop the original categories column from `df`
    df.drop('categories',axis=1,inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filename):
    '''
    INPUT:
    database_filename    - (str) filepath of the database to save the cleaned data

    OUTPUT:
    None
    '''
    
    engine = create_engine('sqlite:///{:}'.format(database_filename))
    df.to_sql('messages', engine, index=False, if_exists='replace')
    


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