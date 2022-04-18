import pickle

with open("dataset.pickle", "rb") as fp:
    data = pickle.load(fp)
    fp.close()

with open(r"word_to_root.pickle", "rb") as vocab_file:
    word_to_root = pickle.load(vocab_file)
    vocab_file.close()

with open(r"words.pickle", "rb") as vocab_file:
    root_to_first = pickle.load(vocab_file)
    vocab_file.close()

with open("train_data.pickle", "rb") as fp:
    train_data = pickle.load(fp)
    fp.close()
history_length = 5
train = []


def get_wids(words):
    wids = []
    for token in words:
        try:
            word_root = word_to_root[token]
            first_word = root_to_first[word_root]
            token = f"/c/en/{first_word}"
            wid = data["word_to_id"][token]
            
            wids.append(wid)
        except KeyError as ke:
            print(ke)
    return wids



for conv_id, conversation in train_data.items():
    initiatorWorkerId = conversation['initiatorWorkerId']
    respondentWorkerId = conversation['respondentWorkerId']
    initiatorContext = []
    initiatorContextId = []
    entityContext = set()
    conv_data = []
    for message in conversation["conversation"]:
        workerId = message["senderWorkerId"]
        if workerId == initiatorWorkerId:
            messageContext = initiatorContext[-history_length+1 if len(initiatorContext) >= history_length else 0:] + [message["tokens"]]
            wids = get_wids(message["tokens"])
            messageContextId = initiatorContextId[-history_length+1 if len(initiatorContext) >= history_length else 0:] + [wids]
            for mid in message["movies"]:
                if int(mid[1:]) not in data["movie_ids"]:
                    continue
                eid = data["movie_ids"][int(mid[1:])]
                if eid not in data["entity_to_id"]:
                    continue
                entityContext.add(eid)
            message_entities = [data["entity_to_id"][ent] for ent in message["entities"] if ent in data["entity_to_id"]]
            entityContext.update(message_entities)
            conv_data.append({
                "context": messageContext,
                "context_ids": messageContextId,
                "entities": entityContext.copy()
            })
            initiatorContext.append(message["tokens"])
            initiatorContextId.append(wids)
        else:
            if not conv_data:
                continue
            conv_data[-1]["response"] = message["tokens"]
    train += conv_data
print()
