import json
import pickle
import time

import requests
from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker

with open("relation_to_id.pickle", "rb") as fp:
    relation_to_id = pickle.load(fp)
    fp.close()

with open("dataset.pickle", "rb") as fp:
    data = pickle.load(fp)
    fp.close()

label_predicates = {"country", "language", "director", "starring", "distributor", "cinematography", "music",
                    "editing", "studio", "writer", "screenplay", "released", "year", "genre", "artist", "story",
                    "company", "productionCompanies", "producers", "editor", "executiveProducer", "composer",
                    "followedBy", "network", "narrator", "creator", "album", "publisher", "author", "owner",
                    "developer", "successor", "industry", "founder"}

predicate_to_skip = {f"http://dbpedia.org/property/{relation}"
                     for relation in relation_to_id
                     if relation not in label_predicates}






kg = KG("https://dbpedia.org/sparql", is_remote=True, skip_predicates=predicate_to_skip)

transformer = RDF2VecTransformer(
    Word2Vec(epochs=10),
    walkers=[RandomWalker(4, 10, with_reverse=False, n_jobs=2)],
    verbose=1
)

result = transformer.fit_transform(kg, filtered_entities)
print()
