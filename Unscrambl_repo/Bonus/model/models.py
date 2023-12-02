from datetime import datetime
from pydantic import BaseModel, Field

class Input(BaseModel):
    transactionID: str = Field(..., title="Unique identifier for transaction", max_length=20)
    dateTime: datetime = Field(..., title="Date and time of transaction")
    customerID: str = Field(..., title="Unique identifier for customer", max_length=20)
    transactionAmount: float = Field(..., title="Amount of transaction", ge=0)

class Output(BaseModel):
    isFraud: bool
    prediction: float
    executionTimeMS: float