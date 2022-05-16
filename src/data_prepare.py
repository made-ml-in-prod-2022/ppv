"""Module for data preparing. 
   Clean data; Feature generation; Train-Test split"""

from argparse import ArgumentParser

import pandas as pd
from sklearn.model_selection import train_test_split


def data_prepare(input_data_path, split):
    """Function prepare dataset for training"""

    train_size = split.train_size = 0.8
    test_size = 1 - train_size
    
    train_dataset_path = 'data/training/train.csv'
    test_dataset_path = 'data/training/test.csv'

    df = pd.read_csv(input_data_path)
    train_df, test_df = train_test_split(df, train_size=train_size, random_state=split.random_state)
    
    print(f'Data from {input_data_path} prepared and splitted to train and test parts ({train_size:4.2f} and {test_size:4.2f} respectively)')
    print(f'Train part was saved to {train_dataset_path}')
    print(f'Test part was saved to {test_dataset_path}')

    train_df.to_csv(train_dataset_path, index=False)
    test_df.to_csv(test_dataset_path, index=False)


if __name__ == '__main__':
    main()
