import json
import pickle
from ast import literal_eval

import pandas as pd
import requests

with open("train_data.pickle", "rb") as fp:
    train_data = pickle.load(fp)
    fp.close()

with open("word_kg.pickle", "rb") as fp:
    word_kg = pickle.load(fp)
    fp.close()

movie_kg = pd.read_csv("movie_kg.csv")

word_kg_clean = []
word_to_id = {}
word_relation_to_id = {}
for e1, r, e2 in word_kg:
    if e1 in word_to_id:
        e1id = word_to_id[e1]
    else:
        e1id = len(word_to_id) + 1
        word_to_id[e1] = e1id
    if e2 in word_to_id:
        e2id = word_to_id[e2]
    else:
        e2id = len(word_to_id) + 1
        word_to_id[e2] = e2id
    if r in word_relation_to_id:
        rid = word_relation_to_id[r]
    else:
        rid = len(word_relation_to_id) + 1
        word_relation_to_id[r] = rid
    word_kg_clean.append((e1id, rid, e2id))
#################### Entity Graph #################
movie_ids = {}
entity_kg_clean = []
entity_to_id = {}
relation_to_id = {
    "year": 1,
    "director": 2,
    "writer": 3,
    "distributor": 4,
    "country": 5,
    "language": 6,
    "producer": 7,
    "music": 8,
    "starring": 9
}
relation_counts = {}
movie_entities = []


def add_edges(entity, is_movie=True):
    if entity in entity_to_id and not is_movie:
        return
    try:
        eid = len(entity_to_id) + 1
        entity_to_id[entity] = eid
        entity_url = entity.replace("resource", "data") + ".json"
        obj = requests.get(entity_url)
        content = json.loads(obj.content)
        properties = content.get(entity, {})
        for property, data in properties.items():
            if "property" in property and "wiki" not in property:
                relation = property.split("/")[-1]
                if relation in ["name", "runtime"]:
                    continue
                if relation in relation_to_id:
                    rid = relation_to_id[relation]
                else:
                    rid = len(relation_to_id) + 1
                    relation_to_id[relation] = rid
                if relation not in relation_counts:
                    relation_counts[relation] = 0
                relation_counts[relation] += 1
                for data_dict in data:
                    e2 = data_dict['value']
                    if e2 in entity_to_id:
                        e2id = entity_to_id[e2]
                    else:
                        e2id = len(entity_to_id) + 1
                        entity_to_id[e2] = e2id
                    edge = (eid, rid, e2id)
                    if edge not in entity_kg_clean:
                        entity_kg_clean.append(edge)
        return eid
    except Exception as e:
        print(e)


def add_movie_edges(row, relation):
    mid = row["mid"]
    movie_id = len(entity_to_id) + 1
    movie_ids[mid] = movie_id
    if relation == "year":
        year = row[relation]
        if year > 0:
            year = int(year)
            edge = (movie_id, relation_to_id["year"], year)
            if edge not in entity_kg_clean:
                if relation not in relation_counts:
                    relation_counts[relation] = 0
                relation_counts[relation] += 1
                entity_kg_clean.append(edge)
    else:
        entities = literal_eval(row[relation])
        for entity in entities:
            if entity in entity_to_id:
                eid = entity_to_id[entity]
            else:
                eid = len(entity_to_id) + 1
                entity_to_id[entity] = eid
            edge = (mid, relation_to_id[relation], eid)
            if edge not in entity_kg_clean:
                if relation not in relation_counts:
                    relation_counts[relation] = 0
                relation_counts[relation] += 1
                entity_kg_clean.append(edge)


for (idx, row) in movie_kg.iterrows():
    row = dict(row)
    movie_URIs = literal_eval(row["movie"])
    if len(movie_URIs) < 1:
        movie = row["movie_name"]
        for relation in ["year", "director", "writer", "distributor", "country", "language", "producer", "music",
                         "starring"]:
            add_movie_edges(row, relation)
    else:
        for movie in movie_URIs:
            if movie in movie_entities:
                continue
            movie_entities.append(movie)
            mid = row["mid"]
            movie_id = add_edges(movie)
            if movie_id is None:
                continue
            movie_ids[mid] = movie_id
for conversation_id, conversation in train_data.items():
    for message in conversation["conversation"]:
        for entity in message["entities"]:
            add_edges(entity, False)

data_to_save = [entity_kg_clean, entity_to_id, movie_ids, relation_counts, relation_to_id, word_kg_clean,
                word_relation_to_id, word_to_id]
data_to_name = ["entity_kg_clean", "entity_to_id", "movie_ids", "relation_counts", "relation_to_id", "word_kg_clean",
                "word_relation_to_id", "word_to_id"]

for data, filename in zip(data_to_save, data_to_name):
    filename += ".pickle"
    with open(filename, "wb") as fp:
        pickle.dump(data, fp)
        fp.close()
