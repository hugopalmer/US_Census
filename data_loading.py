import numpy as np
import pandas as pd
import warnings

'''
This file wraps up several function useful to load the dataset US Census
The main function is load_data(train_or_test)
'''


def column_to_bool(df, col, true_value, false_value):
    dict_map = {false_value: False, true_value: True}
    df[col] = df[col].map(dict_map).astype(bool)
    return df


def set_NA(df, set_NA):
    for NA_like in set_NA:
        df.replace(to_replace=NA_like, value='NaN', inplace=True)
    return df

def load_data(train_or_test, high_income_as_bool=False):
    """
    returns a dataframe prepared with the proper header and dtypes
    """

    if train_or_test == 'train':
        filename = 'census_income_learn.csv'
    elif train_or_test == 'test':
        filename = 'census_income_test.csv'
    else:
        return -1  # wrong argument passed

    # open the csv file that contains no header
    data = pd.read_csv(filename, header=None, skipinitialspace=True)
    # , na_values=["Not in universe or children","Not in universe under 1 year old""Not in universe", 'nan', 'NaN', "?"])

    # remove the column '| instance weight' which is at position 24, as specified
    data_columns = data.columns
    mask = np.ones(data_columns.shape, dtype=bool)
    mask[24] = 0  # 24 is the index of the column we want to remove
    data = data[data_columns[mask]]

    # read from the metadata file the column names
    filename_metadata = 'census_income_metadata.txt'
    cols_read = pd.read_csv(filename_metadata, sep=':', skiprows=141, usecols=[0], comment='|', header=None, dtype=str)

    # remove the 24th column which is '| instance weight'
    # and add 'high_income' (categorical variable to be predicted)
    true_cols = cols_read[0].drop(24).append(pd.Series(['high_income'], index=[41]))
    data.columns = true_cols

    # Certain columns contain numerical data at this stage but should be considered as categorical data:
    cols_to_categorical = ['own business or self employed', 'year', 'detailed occupation recode',
                           'major occupation code',
                           'detailed industry recode', 'major industry code', 'veterans benefits']
    for col in cols_to_categorical:
        data[col] = data[col].astype(str)

    # this column contains a special value that should be considered as nan
    # data["live in this house 1 year ago"].replace(to_replace="Not in universe under 1 year old", value='NaN', inplace=True)
    #data.replace({'Not in universe or children': 'NaN', 'Not in universe under 1 year old': 'NaN'}, regex=True)
    # data.replace({'Not in universe': 'NaN', '?': 'NaN', 'nan': 'NaN'}, regex=True)



    NA_like_list = ["Not in universe or children", "Not in universe under 1 year old", "Not in universe", "?",
                    "Do not know", "nan", "NA", "All other", "Other", "Other service","Not identifiable", "Nonfiler"]
    data = set_NA(data, NA_like_list)

    # some columns may be interpreted as boolean, because they are categorical with only two categories
    data = column_to_bool(data, "fill inc questionnaire for veteran's admin", true_value='Yes', false_value='No')
    data = column_to_bool(data, "member of a labor union", true_value='Yes', false_value='No')
    data = column_to_bool(data, "migration prev res in sunbelt", true_value='Yes', false_value='No')
    data = column_to_bool(data, "live in this house 1 year ago", true_value='Yes', false_value='No')

    #for the data analysis, we prefer to keep the explicit values '50000+.' and '- 50000.': high_income_as_bool= False
    #for the prediction part, we set them to Ture and False to ease their manipulation
    if high_income_as_bool:
        data = column_to_bool(data, "high_income", true_value='50000+.', false_value='- 50000.')

    return data


def get_train_test_sets(data_train, data_test):
    """
    takes the train and test pandas dataframes
    converts the categorical variables to dummy indicators (binaries)

    The main difficulty is that, to use pd.get_dummies(), we have to convert the objects dtypes of the dataframes
    to category dtype, specifying the categories of the train set each time.

    :return: train and test sets in the shape of np.array of integer dtype: X_tr, y_tr, X_te, y_te
    """

    # For both train and test sets, for each column, we only keep the categories present at least once in the train set
    # indeed, if for example there is a special value in the test set that is never encountered in the train set,
    # the ML algos won't be able to catch the information that was not in the train set
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for col in data_test.columns:
            if data_test[col].dtype == 'object':
                data_test[col] = data_test[col].astype('category', categories=np.array(data_train[col].unique()))

        for col in data_train.columns:
            if data_train[col].dtype == 'object':
                data_train[col] = data_train[col].astype('category', categories=np.array(data_train[col].unique()))

        # Call pd.get_dummies that converts very efficiently categorical columns to one-hot binaries
        df_train = pd.get_dummies(data_train)
        df_test = pd.get_dummies(data_test)

    # check if all columns are equal. Should not print anything
    for i in range(len(df_train.columns)):
        if df_train.columns[i] != df_test.columns[i]:
            raise NameError("The columns of the dataframes do not match.")

    X_tr = df_train.drop('high_income', axis=1).as_matrix().astype(np.float64)
    X_te = df_test.drop('high_income', axis=1).as_matrix().astype(np.float64)
    y_tr = df_train['high_income'].as_matrix().astype(int)
    y_te = df_test['high_income'].as_matrix().astype(int)

    return [X_tr, y_tr, X_te, y_te, df_train.columns]


def normalize_continuous(data, data_test, numerical_cols):
    for col in numerical_cols:
        data[col] = data[col].astype(np.float64)
        data[col]=data[col]/max(data[col])
        data_test[col] = data_test[col].astype('float')
        data_test[col] = data_test[col] / max(data_test[col])
    return [data, data_test]


#when ploting some categories, we prefer not to display more than 20 letters on the labels
def truncate(value, size=20):
    if type(value) != str:
        return value
    if len(value)>size:
        return value[:size]
    else:
        return value