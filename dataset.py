import pandas as pd
from datasets import Dataset

dataset = [
    {
        "text": """
    Transaction Successful 
    12:55 pm on 02 Oct 2024

    Paid to 
    DL D R Distributors Private Limited 
    drdistributors@citibank 
    81,574

    Transfer Details
    Transaction ID 
    T2410021255005658879846

    Debited from 
    S XXXXXXXXXXXX2080
    UTR: 427649543696 
    81,574

    Powered by 
    UPI YES BANK 
    UNIFIED PAYMENTS INTERFACE
    """,
        "transaction_id": "T2410021255005658879846",
        "amount": "81574",
        "account_number": "2080",
        "ifsc_code": "",
        "time": "2024-10-02 12:55:00"
    },
    {
        "text": """
Paytm 

PAYMENT SUCCESSFUL 

2,365 

Rupees Two Thousand Three Hundred Sixty Five Only 

To: D R Distributors Private Limited 
UPI ID: drdistributors@citibank 

From: Moeen Khan 
Punjab National Bank - 7968 

UPI Ref ID: 427690915725 
10:15 AM, 02 Oct 2024 
    """,
        "transaction_id": "427690915725",
        "amount": "2365",
        "account_number": "7968",
        "ifsc_code": "",
        "time": "2024-10-02 10:15:00"
    },
    {
        "text": """
Messages • VD-IDFCFB • now

Your A/C XXXXX783558 is debited by INR 3,294.00 on 01/10/24 20:53. New Bal: INR 6,41,576.25. Call us on 180010888 for dispute. Team IDFC FIRST Bank

Mark as read

3,294

01 Oct, 08:57 PM

Ref. No: XX 2944

Check Balance Share Pay Again

Paytm Money

360 0
Platform Fee for Life

300 0
AMC for Life

20 0
Brokerage*

Charges Slashed for You!
Claim Now
    """,
        "transaction_id": "2944",
        "amount": "3294.00",
        "account_number": "783558",
        "ifsc_code": "",
        "time": "2024-10-01 20:53:00"
    },
    {
        "text": """
Send Money

Your request for Send Money transaction of 10858.00 has been accepted. Request you to check the status of your transaction in 'Transaction History'

Reference Number	427585446166

From	9560952931@hdfcbank 
           HDFC 50200087604929
To	D R Distributors Pvt Ltd
           0712866012 
Amount	10,858.00
Date	Tuesday, 1 Oct 2024 
Remarks	Medicines supplier

Share  Download
Add to favourites

DONE
GO TO OVERVIEW
""",
        "transaction_id": "427585446166",
        "amount": "10858.00",
        "account_number": "50200087604929",
        "ifsc_code": "",
        "time": "2024-10-01 00:00:00"
    },
    {
        "text": """
WELCOME GAURAV GARG
Last Log in 01/01/2024 7:08 PM IST

To Other Bank (NEFT)

Reference Number
N275243299738075
From Account
50200080706992
Beneficiary Name
DR DISTRIBUTORS PVT LTD
Beneficiary IFSC Code
CITI0000002
Beneficiary Account Number/Credit Card Number
0712866012
Bank Name
CITI BANK
Transfer Amount
26,198.00
Transfer Description
bill no 375534

realme 11 Pro 5G
lonely planet
""",
        "transaction_id": "N275243299738075",
        "amount": "26198.00",
        "account_number": "50200080706992",
        "ifsc_code": "CITI0000002",
        "time": "2024-01-01 19:08:00"
    }
]

# Convert the list of dictionaries to a pandas DataFrame
df = pd.DataFrame(dataset)

# Convert to a Hugging Face Dataset
hf_dataset = Dataset.from_pandas(df)

# Split the dataset into training and validation sets
hf_dataset = hf_dataset.train_test_split(test_size=0.2)

print(hf_dataset)