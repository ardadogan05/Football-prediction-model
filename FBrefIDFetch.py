import requests
import os
from dotenv import load_dotenv #to keep API key hidden

load_dotenv()

BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")

def fetchID(query):
    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "X-Subscription-Token": BRAVE_API_KEY,
    }
    params = {
        "q": query,
        "count": 1,
    }

    response = requests.get(url, headers=headers, params=params)
    jsonResponse = response.json()
    #prettyJson = json.dumps(jsonResponse, indent = 4) #make the response readable, in order to pick out the part that we need.
    FBrefURL = jsonResponse["web"]["results"][0]["url"]
    #ensuring that we remove empty spaces to help find id and name
    parts = []
    for part in FBrefURL.split('/'):
        if part.strip() != "":
            parts.append(part)
        else:
            continue
    
    FBrefID = parts[4]
    nameList = parts[-1].split('-')
    #to handle instances like "Manchester-City-stats" and "Southampton-stats"
    if len(nameList) == 4:
        FBrefName = nameList[0] + "-" + nameList[1] + "-" + nameList[2]
    elif len(nameList) == 3:
        FBrefName = nameList[0] + "-" + nameList[1]
    elif len(nameList) == 2:
        FBrefName = nameList[0]
    elif len(nameList) == 4:
        FBrefName = nameList[0] + "-" + nameList[1] + "-" + nameList[2]
    elif len(nameList) == 5:
        FBrefName = nameList[0] + "-" + nameList[1] + "-" + nameList[2] + "-" + nameList[3]
    else:
        raise ValueError(f"Unexpected name format in URL: '{parts[-1]}'")
    
    return FBrefID, FBrefName





