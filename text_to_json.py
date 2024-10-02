import json
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# Load the NER model and create a pipeline
model_name = "finiteautomata/bertweet-base-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")


def extract_entities(text):
    entities = ner_pipeline(text)
    return entities

def process_transaction(text):
    # Initialize the pipeline with your chosen model
    nlp = pipeline("text2text-generation", model="google/flan-t5-base")

    # Prepare the prompt
    prompt = f"""
    Extract the following information from the transaction text and format it as JSON:
    - transaction_id: The unique identifier for the transaction (look for ID, REF, or UTR)
    - amount: The transaction amount (look for INR or Rs followed by a number)
    - account_number: The account number (usually masked, like XXXX1234)
    - ifsc_code: The IFSC code (format: 4 letters, 0, 6 alphanumeric characters)
    - time: The date and time of the transaction

    Transaction text:
    {text}

    JSON output:
    """

    # Generate the response
    response = nlp(prompt, max_length=512, num_return_sequences=1)[0]['generated_text']

    # Parse the generated JSON
    try:
        result = eval(response)
    except:
        # If parsing fails, return an empty dictionary
        result = {
            "transaction_id": "",
            "amount": "",
            "account_number": "",
            "ifsc_code": "",
            "time": ""
        }

    return result


# Sample texts
sample_texts = [
    """
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
    """

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
    """

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
    """
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
"""

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
"""
]

# Process each sample text
results = [process_transaction(text) for text in sample_texts]

# Print the results as JSON
print(json.dumps(results, indent=2))
