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
data["id_to_mid"] = {value: key for key, value in data["movie_ids"].items()}
data["entity_to_id"] = {value: key for key, value in data["id_to_entity"].items()}


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


def get_entities(conversation, messages):
    entities = []
    tokens = []
    sentences = []
    for processed_message, message in zip(conversation, messages):
        tokens.append(processed_message["tokens"])
        entities.append(processed_message["entities"])
        sentences.append(message["text"])
    return entities, tokens, sentences


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


columns = ["input_tokens", "input_sentences", "output_entity"]
train_df = pd.DataFrame(columns=columns)


def add_history(tokens, sentences, history=5):
    tokens_with_history = []
    sentences_with_history = []
    for i in range(len(tokens)):
        start = i - history if i >= history else 0
        token_list = []
        sentences_with_history.append(sentences[start:i + 1])
        for message_token in tokens[start:i + 1]:
            token_list += message_token + ["."]
        tokens_with_history.append(token_list)
    return tokens_with_history, sentences_with_history


mid_to_questions = {}
mids = ['115908',
        '184418',
        '125431',
        '88487',
        '104253',
        '181097',
        '125954',
        '93103']

def add_train_data(tokens_with_history, sentences_with_history, all_entities):
    global train_df
    for idx, entity_set in enumerate(all_entities):
        for entity in entity_set:
            _input = tokens_with_history[idx]
            _sentences = sentences_with_history[idx]
            row = [_input, _sentences, entity]
            row = dict(zip(columns, row))
            train_df = train_df.append(row, ignore_index=True)


for conv_id, conversation in train_data.items():
    messages = merge_consecutive_messages(conversation["messages"])
    movies = get_movie_entities(messages)
    entities, tokens, sentences = get_entities(conversation["conversation"], messages)
    all_entities = merge_entity_list(movies, entities)
    tokens_with_history, sentences_with_history = add_history(tokens, sentences)
    add_train_data(tokens_with_history, sentences_with_history, all_entities)
    questions, mentions = conversation["respondentQuestions"], conversation["movieMentions"]
    for key in questions:
        if key in mids:
            mName = mentions[key]
            suggested, seen, liked = tuple(questions[key].values())
            print(key, mName, suggested, seen, liked, sep="\t")
    if train_df.shape[0] == 991:
        break
    print()
    # train_df.to_pickle("train_df.pickle")
sampledf = train_df.iloc[[33, 35, 44, 98, 123]][columns[1:]]
pd.set_option("max_colwidth", None)
print(sampledf)

for mid in mids:
    print(mid_to_questions[str(mid)])
print()
