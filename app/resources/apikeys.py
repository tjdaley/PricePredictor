import os

APIKEYS = {
    "alphavantage": {
        #https://www.alphavantage.co/documentation/#symbolsearch
        "apikey": os.environ["ALPHAVANTAGE_KEY"],
        "doc_url": "https://www.alphavantage.co/documentation/",
        "email": "nottrcp@powerdaley.com",
        "time_series_daily_adjusted_url": \
            "http://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&outputsize=full&symbol={symbol}&apikey={apikey}&datatype=csv"
    }
}