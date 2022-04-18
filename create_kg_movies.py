import json

import pandas as pd
import requests
movie_kg = pd.read_pickle("./movie_kg.pkl")
resource = movie_kg.iloc[0].movie[0]
resource = resource.replace("resource", "data") + ".json"
data = requests.get(resource)
content = json.loads(data.content)
print()