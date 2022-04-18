import pickle

import pandas as pd

with open("entity_to_embedding.pickle", "rb") as fp:
    entity_to_embedding = pickle.load(fp)
    fp.close()
with open("word_to_embedding.pickle", "rb") as fp:
    word_to_embedding = pickle.load(fp)
    fp.close()

train_df = pd.read_pickle("grouped_train_df.pickle").dropna()


def get_word_embeddings(tokens, embedding_len=300, top=20):
    word_embeddings = []
    for token in tokens:
        if len(word_embeddings) == top * embedding_len:
            break
        if token in word_to_embedding:
            word_embeddings += word_to_embedding[token]
    word_embeddings += [0] * (top * embedding_len - len(word_embeddings))
    return word_embeddings


train_df["input_embeddings"] = [get_word_embeddings(input_tokens) for input_tokens in train_df.input_tokens.values]


def get_entity_embeddings(entities, ranked_output, top=20):
    entity_embeddings = []
    for entity in entities + ranked_output:
        if len(entity_embeddings) == top:
            break
        if entity in entity_to_embedding:
            entity_embeddings.append(entity_to_embedding[entity])
    return entity_embeddings


train_df["output_embeddings"] = [get_entity_embeddings(output_entities, ranked_output) for
                                 output_entities, ranked_output in
                                 train_df[["output_entities", "ranked_output"]].values]

columns = ["input_embedding", "output_embedding"]
train_df_expanded = pd.DataFrame(columns=columns)
for input_embeddings, output_embeddings in train_df[["input_embeddings", "output_embeddings"]].values:
    for output_embedding in output_embeddings:
        row = [input_embeddings, output_embedding]
        row = dict(zip(columns, row))
        train_df_expanded = train_df_expanded.append(row, ignore_index=True)
train_df_expanded.to_pickle("train_df_expanded.pickle")
