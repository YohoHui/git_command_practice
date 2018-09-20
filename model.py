# coding: utf-8

# Import necessary libraries for models
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import pandas as pd
pd.set_option('display.max_columns', 30)
import sys
import boto3
import numpy as np
import pickle

version = "0.0.3"
contributors = [
    "contributor1",
    "contributor2",
    "barry",
    "ADD_YOUR_USERNAME",
    "tianli"
]

# download pre-curated training data

s3_client = boto3.client('s3')

s3_client.download_file(Bucket='nab-training-sandpit', Key='read/X_train.csv', Filename='./data/X_train.csv')
s3_client.download_file(Bucket='nab-training-sandpit', Key='read/y_train.csv', Filename='./data/y_train.csv')
s3_client.download_file(Bucket='nab-training-sandpit', Key='read/X_test.csv', Filename='./data/X_test.csv')
s3_client.download_file(Bucket='nab-training-sandpit', Key='read/y_test.csv', Filename='./data/y_test.csv')

# read in training data 

X_train = pd.read_csv('./data/X_train.csv')
y_train = pd.read_csv('./data/y_train.csv', header=None)
X_test = pd.read_csv('./data/X_test.csv')
y_test = pd.read_csv('./data/y_test.csv', header=None)

# train model
model_lm = LinearRegression()
fit_lm = model_lm.fit(X_train, y_train) 


# show some predictions

try: 
    predictions = model_lm.predict(X_test)
    print(predictions)
    output = pd.DataFrame(predictions)
    output.to_csv('predictions.csv')
finally:
    pass


# do deployment if argument provided

if sys.argv[1:]:
    import pickle

    deployable_model = {
        "model": model_lm,
        "metadata": {
            "version": version,
            "contributors": contributors
        }
    }

    pickle.dump(deployable_model, open('clv_model.pickle', 'wb'))

    # upload to s3
    s3_client.upload_file(Filename='clv_model.pickle', 
                        Bucket='nab-training-sandpit',
                        Key='write/clv_model.pickle',
                        ExtraArgs={"ServerSideEncryption": "AES256"})

