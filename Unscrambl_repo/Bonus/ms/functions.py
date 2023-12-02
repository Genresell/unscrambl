import pandas as pd
from ms import model
from model.config import FRAUD_THRESHOLD
from model.models import Output
import time

def isWeekend(dateTime):
    weekday = dateTime.weekday()
    isWeekend = weekday>=5
    
    return int(isWeekend)


def isNight(dateTime):
    transaction_hour = dateTime.hour
    isNight = transaction_hour<=6 or transaction_hour>18
    
    return int(isNight)


#To do: perform lookup for historical data
def getCustomerSpendingBehaviourFeatures(customerTransactions, windowsSizeInDays=[1,7,30]):
    # Let us first order transactions chronologically
    customerTransactions=customerTransactions.sort_values('dateTime')
    
    # The transaction date and time is set as the index, which will allow the use of the rolling function 
    customerTransactions.index=customerTransactions.dateTime
    
    # For each window size
    for windowSize in windowsSizeInDays:
        
        # Compute the sum of the transaction amounts and the number of transactions for the given window size
        sumAmountTransactionWindow=customerTransactions['transactionAmount'].rolling(str(windowSize)+'d').sum()
        numberTransactionWindow=customerTransactions['transactionAmount'].rolling(str(windowSize)+'d').count()
    
        # Compute the average transaction amount for the given window size
        # numberTransactionWindow is always >0 since current transaction is always included
        avgAmtTransactionWindow=sumAmountTransactionWindow/numberTransactionWindow
    
        # Save feature values
        customerTransactions['customerIDNBTransaction'+str(windowSize)+'dayWindow']=list(numberTransactionWindow)
        customerTransactions['customerIDAvgAmt'+str(windowSize)+'dayWindow']=list(avgAmtTransactionWindow)
    
    # Reindex according to transaction IDs
    customerTransactions.index=customerTransactions.transactionID
        
    # And return the dataframe with the new features
    return customerTransactions


def predict(X, model):
    start_time = time.time()
    prediction = model.predict_proba(X)[:,1]
    execution_time = (time.time()-start_time) * 1000
    return prediction, execution_time


def get_model_response(input):
    transaction_df = pd.json_normalize(input.__dict__)

    transaction_df['dateTime'] = pd.to_datetime(transaction_df['dateTime'])
    transaction_df['transactionDuringWeekend']=transaction_df.dateTime.apply(isWeekend)
    transaction_df['transactionDuringNight']=transaction_df.dateTime.apply(isNight)
    transaction_df=transaction_df.groupby('customerID').apply(lambda x: getCustomerSpendingBehaviourFeatures(x, windowsSizeInDays=[1,7,30]))
    transaction_df=transaction_df.sort_values('dateTime').reset_index(drop=True)

    input_features=['transactionAmount', 'transactionDuringWeekend', 'transactionDuringNight', 'customerIDNBTransaction1dayWindow', 'customerIDAvgAmt1dayWindow', 
    'customerIDNBTransaction7dayWindow', 'customerIDAvgAmt7dayWindow', 'customerIDNBTransaction30dayWindow', 'customerIDAvgAmt30dayWindow']

    prediction, execution_time = predict(transaction_df[input_features], model)
    if prediction >= FRAUD_THRESHOLD:
        label = True
    else:
        label = False

    result = Output(
                    isFraud=label, 
                    prediction= prediction,
                    executionTimeMS= execution_time
                    )
    
    return result