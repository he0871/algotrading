import requests

symbols = "TSLA"

url = f"https://data.alpaca.markets/v2/stocks/auctions?symbols={symbols}&start=2024-01-03T00%3A00%3A00Z&end=2024-01-04T00%3A00%3A00Z&limit=1000&feed=sip&sort=asc"


headers = {
    "accept": "application/json",
    "APCA-API-KEY-ID": "PKP6BC14573XB64GYB78",
    "APCA-API-SECRET-KEY": "hfD0n2ORbvpucdMNZj4tHfRlXwgUvrREMaFlnIMr"
}

response = requests.get(url, headers=headers)

print(response.text)
