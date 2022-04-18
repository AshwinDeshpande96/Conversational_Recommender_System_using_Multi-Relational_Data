import pickle

num_history_utterances = 5

data_to_read = []
data_to_name = ["entity_kg_clean", "entity_to_id", "movie_ids", "relation_counts", "relation_to_id", "word_kg_clean",
                "word_relation_to_id", "word_to_id"]

for i, filename in enumerate(data_to_name):
    filename += ".pickle"
    with open(filename, "rb") as fp:
        data_to_read.append(pickle.load(fp))
        fp.close()

dataset = dict(zip(data_to_name, data_to_read))
relation_counts = sorted(dataset["relation_counts"].items(), key=lambda item: item[1], reverse=True)
dataset["id_to_entity"] = {value: key for key, value in dataset["entity_to_id"].items()}
dataset["id_to_relation"] = {value: key for key, value in dataset["relation_to_id"].items()}
relations_requested = {"country", "language", "director", "starring", "distributor", "cinematography", "music",
                       "editing", "studio", "writer", "screenplay", "released", "year", "genre", "artist", "story",
                       "company", "productionCompanies", "producers", "editor", "executiveProducer", "composer",
                       "followedBy", "network", "narrator", "creator", "album", "publisher", "author", "owner",
                       "developer", "successor", "industry", "founder", }
# for relation, rcount in relation_counts:
#     rid = dataset["relation_to_id"][relation]
#     for e1id, rel_id, e2id in dataset["entity_kg_clean"]:
#         if e1id not in dataset["id_to_entity"]:
#             # print("Problem")
#             continue
#         e1 = dataset['id_to_entity'][e1id]
#         if e2id not in dataset["id_to_entity"]:
#             # print("Problem")
#             continue
#         e2 = dataset['id_to_entity'][e2id]
#         edge = (e1, relation, e2)
#         if rel_id == rid:
#             print("#"*50)
#             print(relation, rcount)
#             print(edge)
#             break
# exit(0)
########################## Remove extra data #########################


entity_kg = []
entities = set()
dataset["entity_to_edge"] = {}
for e1id, rel_id, e2id in dataset["entity_kg_clean"]:
    if dataset["id_to_relation"][rel_id] in relations_requested:
        if e1id not in dataset["entity_to_edge"]:
            dataset["entity_to_edge"][e1id] = set()
        if e2id not in dataset["entity_to_edge"]:
            dataset["entity_to_edge"][e2id] = set()
        if (e1id, rel_id, e2id) not in entity_kg:
            entity_kg.append((e1id, rel_id, e2id))
            dataset["entity_to_edge"][e1id].add(len(entity_kg) - 1)
            dataset["entity_to_edge"][e2id].add(len(entity_kg) - 1)
            entities.update([e1id, e2id])
        else:
            print("problem!")

dataset["entity_kg_clean"] = entity_kg

dataset["relation_to_id"] = {relation: dataset["relation_to_id"][relation] for relation in relations_requested}
dataset["id_to_relation"] = {value: key for key, value in dataset["relation_to_id"].items()}

entities_to_remove = []
for eid in dataset["id_to_entity"]:
    if eid not in entities:
        entities_to_remove.append(eid)

for eid in entities_to_remove:
    # print(dataset["id_to_entity"][eid])
    dataset["id_to_entity"].pop(eid)
dataset["entity_to_id"] = {value: key for key, value in dataset["id_to_entity"].items()}

with open("dataset.pickle", "wb") as fp:
    pickle.dump(dataset, fp)
    fp.close()
print()
