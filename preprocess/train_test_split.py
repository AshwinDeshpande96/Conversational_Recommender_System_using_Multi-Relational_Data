import random

import pandas as pd

kgdf = pd.read_pickle("kgdf.pickle")

kgdf["entity1"] = kgdf.entity1.apply(lambda x: x.strip())
kgdf["relation"] = kgdf.relation.apply(lambda x: x.strip())
kgdf["entity2"] = kgdf.entity2.apply(lambda x: x.strip())

kgdf_clean = kgdf[~((kgdf["entity1"] == "") | (kgdf["relation"] == "") | (kgdf["entity2"] == ""))]


# triplets = kgdf_clean.values

def split_kg(df, test_split=0.3, error=50, max_retries=50):
    df = df.reset_index(drop=True)
    e_to_id = {}
    id_to_e = {}
    total_size = df.shape[0]
    for idx, (e1, e2) in df[["entity1", "entity2"]].iterrows():
        if e1 not in e_to_id:
            e_to_id[e1] = set()
        if e2 not in e_to_id:
            e_to_id[e2] = set()
        if idx > total_size:
            print(idx)
        id_to_e[idx] = [e1, e2]
        e_to_id[e1].add(idx)
        e_to_id[e2].add(idx)

    test_size = test_split * total_size
    train_size = total_size - test_size
    test_ids = set()
    train_ids = set()
    test_entities = set()
    train_entities = set()
    retries = 0
    while len(test_ids) < test_size - error and len(train_ids) < train_size - error:
        if retries > max_retries:
            break
        e = random.choice(list(e_to_id.keys()))
        ids = e_to_id[e]
        entities = set()
        for _id in ids:
            entities.update(id_to_e[_id])
        prob = random.uniform(0, 1)
        if prob < test_split:
            in_train = len(train_entities.intersection(entities)) > 0
            if in_train:
                retries += 1
                continue
            retries = 0
            test_entities.update(entities)
            test_ids.update(ids)
            e_to_id.pop(e)
        else:
            in_test = len(test_entities.intersection(entities)) > 0
            if in_test:
                retries += 1
                continue
            retries = 0
            train_entities.update(entities)
            train_ids.update(ids)
            e_to_id.pop(e)
    remaining = list(e_to_id.keys())
    for e in remaining:
        ids = e_to_id[e]
        entities = set()
        for _id in ids:
            entities.update(id_to_e[_id])
        in_train = len(train_entities.intersection(entities)) > 0
        if in_train:
            continue
        test_entities.update(entities)
        test_ids.update(ids)
        e_to_id.pop(e)
    remaining = list(e_to_id.keys())
    for e in remaining:
        ids = e_to_id[e]
        entities = set()
        for _id in ids:
            entities.update(id_to_e[_id])
        in_train = len(train_entities.intersection(entities)) > 0
        in_test = len(test_entities.intersection(entities)) > 0
        if in_train and in_test:
            continue
        elif in_train:
            train_entities.update(entities)
            train_ids.update(ids)
            e_to_id.pop(e)
        else:
            test_entities.update(entities)
            test_ids.update(ids)
            e_to_id.pop(e)
    is_overlap = train_entities.intersection(test_entities)
    is_out_of_bounds = [_id for _id in train_ids if _id > total_size]
    is_out_of_bounds = [_id for _id in test_ids if _id > total_size]
    train = df.iloc[list(train_ids)]
    test = df.iloc[list(test_ids)]
    return train, test

from pykeen.pipeline import  pipeline
train, test = split_kg(kgdf_clean)
train.to_pickle("train_kg.pickle")
test.to_pickle("test_kg.pickle")
