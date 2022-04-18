import pickle
import re

import pandas as pd

with open("train_data.pickle", "rb") as fp:
    train_data = pickle.load(fp)
    fp.close()

data = {}
data_required = ["movie_ids", "id_to_entity"]
for key in data_required:
    with open(f"preprocess/final_data/{key}.pickle", "rb") as fp:
        data[key] = pickle.load(fp)
        fp.close()


def merge_consecutive_messages(messages):
    prev_sender = None
    conversation = []
    for message in messages:
        if message['senderWorkerId'] != prev_sender:
            conversation.append({"text": message['text'],
                                 "senderWorkerId": message["senderWorkerId"]})
            prev_sender = message['senderWorkerId']
        else:
            conversation[-1]['text'] += ". " + message["text"]
    return conversation


def get_entities(conversation):
    entities = []
    tokens = []
    for message in conversation:
        tokens.append(message["tokens"])
        entities.append(message["entities"])
    return entities, tokens


def get_movie_entities(messages):
    movie_entities = []
    for message in messages:
        text = message["text"]
        movie_ids = re.findall("\@[0-9]+", text)
        movies = []
        for mid in movie_ids:
            if int(mid[1:]) not in data["movie_ids"]:
                continue
            eid = data["movie_ids"][int(mid[1:])]
            if eid in data["id_to_entity"]:
                movie = data["id_to_entity"][eid]
                movies.append(movie)
        movie_entities.append(movies)
    return movie_entities


def merge_entity_list(movies, entities):
    new_entities = []
    for movie_list, entity_list in zip(movies, entities):
        new_entities.append(set(movie_list + entity_list))
    return new_entities


columns = ["input_tokens", "output_entity"]
train_df = pd.DataFrame(columns=columns)


def add_history(tokens, history=5):
    tokens_with_history = []
    for i in range(len(tokens)):
        start = i - history if i >= history else 0
        token_list = []
        for message_token in tokens[start:i + 1]:
            token_list += message_token + ["."]
        tokens_with_history.append(token_list)
    return tokens_with_history


def add_train_data(tokens_with_history, all_entities):
    global train_df
    for idx, entity_set in enumerate(all_entities):
        for entity in entity_set:
            _input = tokens_with_history[idx]
            row = [_input, entity]
            row = dict(zip(columns, row))
            train_df = train_df.append(row, ignore_index=True)


for conv_id, conversation in train_data.items():
    messages = merge_consecutive_messages(conversation["messages"])
    movies = get_movie_entities(messages)
    entities, tokens = get_entities(conversation["conversation"])
    all_entities = merge_entity_list(movies, entities)
    tokens_with_history = add_history(tokens)
    add_train_data(tokens_with_history, all_entities)
    train_df.to_pickle("train_df.pickle")

