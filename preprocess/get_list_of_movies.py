import json
import pickle

import requests
from tqdm import tqdm

# tf = TriplesFactory.from_path(
#     "preprocess/full_kg.tsv"
# )
#
type_key = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'
# film_types = ['http://dbpedia.org/ontology/Film', 'http://dbpedia.org/ontology/TelevisionShow']
# train_df = pd.read_pickle("train_df.pickle")
# list_of_movies = []
# missed_type = []
# all_entities = set(list(tf.entity_to_id.keys()) + train_df.output_entity.unique().tolist())
# for ent in all_entities:
#     if not isinstance(ent, str):
#         continue
#     if "dbpedia" not in ent:
#         continue
#     try:
#         entity_url = ent.replace("resource", "data") + ".json"
#         obj = requests.get(entity_url)
#         content = json.loads(obj.content)
#         if ent in content and type_key in content[ent]:
#             content = str(content[ent][type_key])
#             for accepted_type in film_types:
#                 if accepted_type in content:
#                     list_of_movies.append(ent)
#                     break
#     except Exception as e:
#         print(e)
# with open("list_of_movies.pickle", "wb") as fp:
#     pickle.dump(list_of_movies, fp)
#     fp.close()
list_of_movies_clean = []
with open("list_of_movies.pickle", "rb") as fp:
    list_of_movies = pickle.load(fp)
    fp.close()


def is_film(content):
    if "http://dbpedia.org/ontology/Animal" in content:
        return False
    elif "http://dbpedia.org/ontology/Person" in content:
        return False
    elif "http://dbpedia.org/ontology/Species" in content:
        return False
    elif "http://dbpedia.org/ontology/Film" in content:
        return True
    elif "http://dbpedia.org/ontology/TelevisionShow" in content:
        return True
    else:
        return False


for ent in tqdm(list_of_movies):
    try:
        entity_url = ent.replace("resource", "data") + ".json"
        obj = requests.get(entity_url)
        content = json.loads(obj.content)
        if ent in content and type_key in content[ent]:
            content = str(content[ent][type_key])
            if is_film(content):
                list_of_movies_clean.append(ent)
    except Exception as e:
        print(e)
print()
with open("list_of_movies.pickle", "wb") as fp:
    pickle.dump(list_of_movies_clean, fp)
    fp.close()
