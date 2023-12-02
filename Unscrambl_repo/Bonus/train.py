import pandas as pd
import numpy as np

import datetime
import time
import xgboost as xgboost
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
import joblib
import gzip

transactions_df = pd.read_csv('transactions.csv')
transactions_df['TX_DATETIME'] = pd.to_datetime(transactions_df['TX_DATETIME'])

start_date_training = datetime.datetime.strptime("2021-01-01", "%Y-%m-%d")
delta_train = delta_delay = delta_test = 7

def get_train_test_set(transactions_df,
                       start_date_training,
                       delta_train=7,delta_delay=7,delta_test=7):
    
    # Get the training set data
    train_df = transactions_df[(transactions_df.TX_DATETIME>=start_date_training) &
                               (transactions_df.TX_DATETIME<start_date_training+datetime.timedelta(days=delta_train))]
    
    # Get the test set data
    test_df = []
    
    # Note: Cards known to be compromised after the delay period are removed from the test set
    # That is, for each test day, all frauds known at (test_day-delay_period) are removed
    
    # First, get known defrauded customers from the training set
    known_defrauded_customers = set(train_df[train_df.TX_FRAUD==1].CUSTOMER_ID)
    
    # Get the relative starting day of training set (easier than TX_DATETIME to collect test data)
    start_transaction_time_days_training = train_df.TX_TIME_DAYS.min()
    
    # Then, for each day of the test set
    for day in range(delta_test):
    
        # Get test data for that day
        test_df_day = transactions_df[transactions_df.TX_TIME_DAYS==start_transaction_time_days_training+
                                                                    delta_train+delta_delay+
                                                                    day]
        
        # Compromised cards from that test day, minus the delay period, are added to the pool of known defrauded customers
        test_df_day_delay_period = transactions_df[transactions_df.TX_TIME_DAYS==start_transaction_time_days_training+
                                                                                delta_train+
                                                                                day-1]
        
        new_defrauded_customers = set(test_df_day_delay_period[test_df_day_delay_period.TX_FRAUD==1].CUSTOMER_ID)
        known_defrauded_customers = known_defrauded_customers.union(new_defrauded_customers)
        
        test_df_day = test_df_day[~test_df_day.CUSTOMER_ID.isin(known_defrauded_customers)]
        
        test_df.append(test_df_day)
        
    test_df = pd.concat(test_df)
    
    # Sort data sets by ascending order of transaction ID
    train_df=train_df.sort_values('TRANSACTION_ID')
    test_df=test_df.sort_values('TRANSACTION_ID')
    
    return (train_df, test_df)

transactions_df.rename(columns = {'TX_AMOUNT': 'transactionAmount', 'TX_DURING_WEEKEND': 'transactionDuringWeekend', 'TX_DURING_NIGHT': 'transactionDuringNight', 
'CUSTOMER_ID_NB_TX_1DAY_WINDOW': 'customerIDNBTransaction1dayWindow', 'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW': 'customerIDAvgAmt1dayWindow',
'CUSTOMER_ID_NB_TX_7DAY_WINDOW': 'customerIDNBTransaction7dayWindow', 'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW': 'customerIDAvgAmt7dayWindow',
'CUSTOMER_ID_NB_TX_30DAY_WINDOW': 'customerIDNBTransaction30dayWindow', 'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW': 'customerIDAvgAmt30dayWindow'}, inplace=True)

(train_df, test_df)=get_train_test_set(transactions_df,start_date_training,
                                       delta_train=7,delta_delay=7,delta_test=7)

output_feature="TX_FRAUD"

input_features=['transactionAmount', 'transactionDuringWeekend', 'transactionDuringNight', 'customerIDNBTransaction1dayWindow', 'customerIDAvgAmt1dayWindow', 
    'customerIDNBTransaction7dayWindow', 'customerIDAvgAmt7dayWindow', 'customerIDNBTransaction30dayWindow', 'customerIDAvgAmt30dayWindow']

classifier = xgboost.XGBClassifier()

parameters = {'clf__max_depth':[3], 'clf__n_estimators':[50], 'clf__learning_rate':[0.1],
              'clf__random_state':[0], 'clf__n_jobs':[1], 'clf__verbosity':[0]}

classifier.fit(train_df[input_features], train_df[output_feature])

# Export model
joblib.dump(classifier, gzip.open('model_binary.dat.gz', "wb"))