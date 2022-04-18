import pickle

import pandas as pd
from pykeen.triples import TriplesFactory
# class Node:
#     def __init__(self, ent_name, children=set()):
#         self.children = children
#         self.ent_name = ent_name
from tqdm import tqdm

tf = TriplesFactory.from_path(
    "preprocess/full_kg.tsv"
)

entity_kg = pd.read_csv("preprocess/full_kg.tsv", sep="\t")
train_df = pd.read_pickle("train_df.pickle")
with open(r"list_of_movies.pickle", "rb") as fp:
    list_of_movies = pickle.load(fp)
    fp.close()

all_entities = {}
# www = entity_kg[(entity_kg.entity1 == 'http://dbpedia.org/resource/Wild_Wild_West') | (entity_kg.entity2 == 'http://dbpedia.org/resource/Wild_Wild_West')]

for idx, (e1, e2) in entity_kg[["entity1", "entity2"]].iterrows():
    if e1 not in all_entities:
        all_entities[e1] = set()
    if e2 not in all_entities:
        all_entities[e2] = set()
    all_entities[e1].add(e2)
    all_entities[e2].add(e1)

data = {}
data_required = ["movie_ids", "id_to_entity"]
for key in data_required:
    with open(f"preprocess/final_data/{key}.pickle", "rb") as fp:
        data[key] = pickle.load(fp)
        fp.close()

train_df["key"] = train_df.input_tokens.apply(lambda x: ",".join(x))
grouped_train_df = train_df.groupby("key")["output_entity"].apply(list).reset_index(name='output_entities')

entities_not_found = set()


def get_ranked_output(output_entities, top=100):
    if not isinstance(output_entities, list):
        return
    bfs_list = []
    for ent in set(output_entities):
        if ent in all_entities:
            bfs_list.append(ent)
        else:
            entities_not_found.add(ent)
    ranked_output = []
    i = 0
    while len(ranked_output) != top:
        if i >= len(bfs_list):
            break
        entity = bfs_list[i]
        i += 1
        if entity in list_of_movies and entity not in ranked_output:
            ranked_output.append(entity)
        children = all_entities.get(entity, [])
        for child in children:
            if child not in bfs_list:
                bfs_list.append(child)
    for movie in list_of_movies:
        if movie not in ranked_output:
            ranked_output.append(movie)
    return ranked_output


grouped_train_df["ranked_output"] = None


def get_tokens(x):
    x = x.split(",")
    tokens = ["#"]
    for token in x:
        if token == tokens[-1]:
            continue
        tokens.append(token)
    tokens = tokens[1:]
    if tokens and tokens[0] == ".":
        tokens = tokens[1:]
    return tokens


grouped_train_df["input_tokens"] = grouped_train_df.key.apply(lambda x: get_tokens(x))
grouped_train_df = grouped_train_df.drop(0)
grouped_train_df = grouped_train_df.reset_index()
grouped_train_df = grouped_train_df.drop(columns=["key", "index"])[["input_tokens", "output_entities", "ranked_output"]]
for idx, (input_tokens, output_entities, ranked_output) in tqdm(grouped_train_df.iterrows()):
    grouped_train_df.loc[idx, "ranked_output"] = get_ranked_output(output_entities)
    grouped_train_df.to_pickle("grouped_train_df.pickle")
    grouped_train_df.to_csv("grouped_train_df.csv", index=False)
