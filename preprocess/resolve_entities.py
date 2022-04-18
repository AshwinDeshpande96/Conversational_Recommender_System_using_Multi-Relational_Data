import json
import pickle
import time

import requests

with open("dataset.pickle", "rb") as fp:
    dataset = pickle.load(fp)
    fp.close()
all_entities = list(dataset["entity_to_id"].keys())


def get_dbpedia_link(ent):
    time.sleep(1)
    try:
        obj = requests.get(
            f"https://lookup.dbpedia.org/api/search/KeywordSearch?format=JSON&QueryString={ent}")
        data = json.loads(obj.content)
        for doc in data.get("docs", []):
            dbpedia_link = doc.get("resource")
            if dbpedia_link:
                return dbpedia_link[0]
    except Exception as e:
        print(e)


edges_to_be_removed = set()
entities_to_be_removed = set()


def decompound_entities(compounded_entities, eid):

    entities_to_be_removed.add(eid)
    edges_to_be_removed.update(dataset["entity_to_edge"][eid])
    new_entities = []
    for ent in compounded_entities:
        ent = ent.strip()
        dbpedia_link = get_dbpedia_link(ent)
        if dbpedia_link is not None:
            new_entities.append(dbpedia_link)
    for new_ent in new_entities:
        if new_ent not in dataset["entity_to_id"]:
            new_eid = len(dataset["entity_to_id"]) + 1
            dataset["entity_to_id"][new_ent] = new_eid
            dataset["id_to_entity"][new_eid] = new_ent
        else:
            new_eid = dataset["entity_to_id"][new_ent]
        for edge_id in dataset["entity_to_edge"][eid]:
            e1id, rid, e2id = dataset["entity_kg_clean"][edge_id]
            if e1id == eid:
                new_edge = (new_eid, rid, e2id)
            else:
                new_edge = (e1id, rid, new_eid)
            dataset["entity_kg_clean"].append(new_edge)


while all_entities:
    entity = all_entities[0]
    eid = dataset["entity_to_id"][entity]
    all_entities.remove(entity)
    if not isinstance(entity, str):
        for i in dataset["entity_to_edge"][eid]:
            dataset["entity_kg_clean"][i] = None
        continue
    if "dbpedia" in entity:
        continue
    if "," in entity:
        decompound_entities(entity.split(","), eid)
    elif "/" in entity:
        decompound_entities(entity.split("/"), eid)
    else:
        decompound_entities([entity], eid)

for edge_id in edges_to_be_removed:
    dataset["entity_kg_clean"].pop(edge_id)

for eid in entities_to_be_removed:
    entity = dataset["id_to_entity"][eid]
    dataset["id_to_entity"].pop(eid)
    dataset["entity_to_id"].pop(entity)


