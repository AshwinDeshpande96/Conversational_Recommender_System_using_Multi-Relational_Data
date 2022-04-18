#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import pickle

import requests

# In[2]:


keys = ["entity_to_id", "id_to_entity", "entity_kg_clean",
        "id_to_relation", "relation_to_id", "progress",
        "new_id_map", "all_entities"]

# In[3]:


with open("temp.pickle", "rb") as fp:
    dataset = pickle.load(fp)
    fp.close()


# In[4]:


def save():
    data = {}
    for key in keys:
        data[key] = dataset[key].copy()
    #     print(data.keys())
    with open("temp.pickle", "wb") as fp:
        pickle.dump(data, fp)
        fp.close()


# In[5]:


# with open("../dataset2.pickle", "rb") as fp:
#     dataset = pickle.load(fp)
#     fp.close()
# dataset["progress"] = {}
# dataset["new_id_map"] = {}
# dataset["all_entities"] = list(dataset["entity_to_id"].keys())
# save()


# In[6]:


dataset.keys()


# In[7]:


def get_dbpedia_link(ent):
    # time.sleep(1)
    try:
        obj = requests.get(
            f"https://lookup.dbpedia.org/api/search/KeywordSearch?format=JSON&QueryString={ent}")
        data = json.loads(obj.content)
        for doc in data.get("docs", []):
            dbpedia_link = doc.get("resource")
            score = doc.get("score")
            if dbpedia_link and float(score[0]) > 2000:
                #                 print(ent, dbpedia_link, score)
                return dbpedia_link[0]
    except Exception as e:
        print(e)


# In[8]:


def decompound_entities(compounded_entities, eid):
    if eid not in dataset["new_id_map"]:
        dataset["new_id_map"][eid] = set()
    for ent in compounded_entities:
        ent = ent.strip()
        new_ent = get_dbpedia_link(ent)
        if new_ent is not None:
            if new_ent not in dataset["entity_to_id"]:
                new_eid = len(dataset["entity_to_id"]) + 1
                dataset["entity_to_id"][new_ent] = new_eid
                dataset["id_to_entity"][new_eid] = new_ent
            else:
                new_eid = dataset["entity_to_id"][new_ent]
            dataset["new_id_map"][eid].add(new_eid)


# In[9]:


# for entity in tqdm(dataset["all_entities"]):  
#     if entity in dataset["progress"]:
#         continue
#     dataset["progress"][entity] = 1
#     save()
#     eid = dataset["entity_to_id"][entity]
#     if not isinstance(entity, str):
#         dataset["new_id_map"][eid] = set()
#         continue
#     if re.match("[0-9]{4}\-[0-9]{2}\-[0-9]{2}", entity):
#         dataset["new_id_map"][eid] = set()
#         continue
#     if "dbpedia" in entity:
#         continue
#     if "," in entity:
#         decompound_entities(entity.split(","), eid)
#     elif "/" in entity:
#         decompound_entities(entity.split("/"), eid)
#     elif "\n" in entity:
#         decompound_entities(entity.split("\n"), eid)
#     else:
#         decompound_entities([entity], eid)


# In[10]:


print(f"Completed: {len(dataset['progress'])}/{len(dataset['all_entities'])}")
print(len(dataset["new_id_map"]))
for eid in dataset["new_id_map"]:
    if not dataset["new_id_map"][eid]:
        print(dataset["id_to_entity"][eid])
# In[12]:

entity_to_id = {}
id_to_entity = {}
relation_to_id = {}
id_to_relation = {}
entity_kg_clean = set([])
for i in range(len(dataset["entity_kg_clean"])):
    try:
        e1id, rid, e2id = dataset["entity_kg_clean"][i]
        if e1id in dataset["new_id_map"]:
            e1ids = dataset["new_id_map"][e1id]
            e1s = [dataset["id_to_entity"][eid] for eid in e1ids]
        else:
            e1s = [dataset["id_to_entity"][e1id]]
        if e2id in dataset["new_id_map"]:
            e2ids = dataset["new_id_map"][e2id]
            e2s = [dataset["id_to_entity"][eid] for eid in e2ids]
        else:
            e2s = [dataset["id_to_entity"][e2id]]
        r = dataset["id_to_relation"][rid]
        if r in relation_to_id:
            new_rid = relation_to_id[r]
        else:
            new_rid = len(relation_to_id) + 1
            relation_to_id[r] = new_rid
            id_to_relation[new_rid] = r

        for e1 in e1s:
            if e1 in entity_to_id:
                new_e1id = entity_to_id[e1]
            else:
                new_e1id = len(entity_to_id) + 1
                entity_to_id[e1] = new_e1id
                id_to_entity[new_e1id] = e1
            for e2 in e2s:
                if e2 in entity_to_id:
                    new_e2id = entity_to_id[e2]
                else:
                    new_e2id = len(entity_to_id) + 1
                    entity_to_id[e2] = new_e2id
                    id_to_entity[new_e2id] = e2
                entity_kg_clean.add((new_e1id, new_rid, new_e2id))
    except Exception as e:
        print(e)
dataset["entity_to_id"] = entity_to_id
dataset["id_to_entity"] = id_to_entity
dataset["relation_to_id"] = relation_to_id
dataset["id_to_relation"] = id_to_relation
dataset["entity_kg_clean"] = entity_kg_clean
print()
# In[ ]:


with open("../dataset3.pickle", "wb") as fp:
    pickle.dump(dataset, fp)
    fp.close()


# In[ ]:
