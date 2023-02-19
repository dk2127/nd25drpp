# -*- coding: utf-8 -*-
"""
The first part of data pipeline is the Extract, Transform, and Load process. 
Reads the dataset, cleans the data, and then stores it in a SQLite database. 

Created on Feb 2023
@author: DK
"""
# 1 Import libraries
import os
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    This function takes 2 filepaths inputs, loads and merges them.
    Returns the merged dataset as a dataframe

    Parameters:
    messages_filepath, categories_filepath: The files to be loaded and merged.

    Returns:
    dataframe: The dataframe with 2 merged datasets on id
    """
    print("==========<<  Now in function  load_data\n")
    # Load messages dataset
    messages = pd.read_csv(messages_filepath, encoding='latin-1')
    messages.head()

    # Load categories dataset
    categories = pd.read_csv(categories_filepath, encoding='latin-1')
    categories.head()

    # Merge datasets
    df = messages.merge(categories, how='inner', on=["id"])
    df.head()

    return df


def remove_duplicates_from_df(df):
    """
    This function takes a dataframe as input and removes any duplicates from the dataframe.
    Returns the dataframe without duplicates, and message indicating no. of duplicates removed.

    Parameters:
    df (dataframe): The dataframe from which duplicates should be removed.

    Returns:
    dataframe, str: The dataframe with duplicates removed, and message indicating no. of
    duplicates removed
    """
    print("==========<<  Now in function  remove_duplicates_from_df\n")
    original_row_count = df.shape[0]

    df.drop_duplicates(inplace=True)

    new_row_count = df.shape[0]

    duplicates_removed = original_row_count - new_row_count

    return df, f"{duplicates_removed} duplicates removed from the dataframe."


def clean_data(df):
    """
    This function takes a dataframe as input and cleans the data.
    Adds the 36 individual category columns as 0 and 1.
    Returns the dataframe.

    Parameters:
    df (dataframe): The dataframe that needs cleaned up.

    Returns:
    dataframe: The cleaned up dataframe.
    """
    print("==========<<  Now in function  clean_data\n")
    # create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(";", expand=True)
    categories.head()

    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # print(row)
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[0:-2])
    print(category_colnames)

    # rename the columns of `categories`
    categories.columns = category_colnames
    categories.head()

    # Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(
            "str").apply(lambda x: x[-1])
        # convert column from string to numeric
        categories[column] = categories[column].astype("int32")

    categories.head()
    categories.describe()

    # Above result rows, there is one more than expected 0 and 1!
    # May be we have to clean it up later
    categories["related"].unique()

    # Replace categories column in df with new category columns.
    categories.head()
    # drop the original categories column from `df`
    df.drop(["categories"], axis=1, inplace=True)
    df.head()

    # Concatenate df and categories data frames.
    df = pd.concat([df, categories], axis=1)
    df.head()

    # let's check new df summary stats
    df.describe()
    # Let's verify the child_alone has only zeros and then drop the column
    df["child_alone"].unique()

    # drop the column child_alone since has all 0
    df.drop(labels=["child_alone"], axis=1, inplace=True)

    # Let's verify the related column has only 0 and 1
    df["related"].unique()

    # Drop the rows with related = 2. First get the row numbers where related =2
    related2_rownum = df[(df['related'] == 2)].index

    # verify those are the rows with related = 2
    df.iloc[related2_rownum]

    # drop those rows from the dataframe
    df.drop(related2_rownum, inplace=True)

    # Let's verify the related column has only 0 and 1
    df["related"].unique()

    # 6. Remove duplicates.

    # check number of duplicates
    print("Total rows=", df.count())
    print("% duplicate rows=", df.duplicated(subset=None, keep='first').mean())

    # drop duplicate rows
    df, dupknt = remove_duplicates_from_df(df)
    print(dupknt)

    # check number of duplicate rows after
    print("Total rows=", df.count())
    print("% duplicate rows=", df.duplicated(subset=None, keep='first').mean())

    return df


def save_data(df, database_filename):
    """
    This function takes a dataframe as input and saves it to a SQLITE db.
    Adds the 36 individual category columns as 0 and 1.
    Returns the none.

    Parameters:
    df (dataframe): The dataframe that needs loaed.
    database_filename: SQLITE database filename

    Returns:
    None.
    """
    print("==========<<  Now in function  save_data\n")
    # 7. Save the clean dataset into an sqlite database.
    # First check if the DB file exists, and if it does,
    # delete it before creating the SQLite engine.
    if os.path.exists(database_filename):
        print("removing old db", database_filename)
        os.remove(database_filename)

    # Create the SQLite engine
    database_url = 'sqlite:///'+database_filename
    print("database url=", database_url)
    engine = create_engine(database_url, pool_pre_ping=True)

    # copy the df dataframe to SQLITE db with tablename before the .db extention
    tablename = database_filename.split('.')[0]
    print("Saving to tablename=", tablename)
    df.to_sql(tablename, engine, if_exists='replace', index=False)

    # validate the records are wrtten to the database
    query = query = f"SELECT COUNT(*) FROM {tablename}"
    result = engine.execute(query)
    row_count = result.fetchone()[0]
    print('Number of rows written :', row_count)
    print("Dateframe row count was:", df.shape[0])

    # close sql engine
    engine.dispose()

    return None


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
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
