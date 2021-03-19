import pandas as pd
import numpy as np

if __name__ == '__main__':
    df = pd.read_csv('./bank-additional-full.csv', sep=';')

    # Use for loop to handle unknown values in the categorical variables
    for col in df.iloc[:,1:7].columns:
        for i, value in enumerate(df[col].values):
            if df[col][i] == 'unknown':
                df[col][i].replace(df[col][i], df[col].mode()[0])

    # Drop unnecessary columns
    df = df.drop('contact', axis=1)

    # Change duration from seconds to minutes
    months_to_replace = {
        'jan': 1,
        'feb': 2,
        'mar': 3,
        'apr': 4,
        'may': 5,
        'jun': 6,
        'jul': 7,
        'aug': 8,
        'sep': 9,
        'oct': 10,
        'nov': 11,
        'dec': 12
    }

    df['month'] = df['month'].map(months_to_replace)

    # transform binary variable in numeric
    mapping_dict = {
        'yes': 1,
        'no': 0
    }

    df['response'] = df.rename(index="str", columns={'y': "response"})
    df['response'] = df['response'].apply(lambda x: mapping_dict[x])

    # Filtering - delete less than 10 seconds
    duration_10_ = df['duration'] > 10/60
    df['duration'] = df['duration'].drop(df[~duration_10_].index)

    # Drop more columns for machine learning
    drop_columns = ['pdays',
                    'poutcome',
                    'emp.var.rate',
                    'cons.price.idx',
                    'cons.conf.idx',
                    'euribor3m',
                    'nr.employed']

    df = df.drop(columns=drop_columns)

    # Transform cateogrical features
    def education(edu):
        if (edu == 'basic.4y' or edu == 'basic.6y' or edu == 'basic.9y'):
            return 'basic'
        elif edu =='high.school':
            return 'secondary'
        elif (edu =='university.degree' or edu == 'professional.course'):
            return 'tertiary'
        else:
            return 'illiterate'

    df['education'] = df['education'].apply(education)

    # Mapping binary variables into numeric
    df['default'] = df['default'].map(mapping_dict)
    df['housing'] = df['housing'].map(mapping_dict)
    df['loan'] = df['loan'].map(mapping_dict)

    # change dow into numeric
    day_numeric = {
        'mon': 1,
        'tue': 2,
        'wed': 3,
        'thu': 4,
        'fri': 5
    }

    df['day_of_week'] = df['day_of_week'].map(day_numeric)

    # get dummies for categorical variables
    df = pd.get_dummies(df, columns=['job', 'marital', 'education'])

