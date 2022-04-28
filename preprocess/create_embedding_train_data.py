import pickle

import pandas as pd

with open("entity_to_embedding.pickle", "rb") as fp:
    entity_to_embedding = pickle.load(fp)
    fp.close()
with open("word_to_embedding.pickle", "rb") as fp:
    word_to_embedding = pickle.load(fp)
    fp.close()

train_df = pd.read_pickle("grouped_train_df.pickle").dropna()


def get_word_embeddings(tokens, embedding_len=300, top=300):
    word_embeddings = []
    words = []
    for token in tokens:
        if len(word_embeddings) == top * embedding_len:
            break
        if token in word_to_embedding:
            word_embeddings.append(word_to_embedding[token])
            words.append(token)
        else:
            words.append('[UNK]')
            word_embeddings.append([[0] * embedding_len])
    word_embeddings += [[0] * embedding_len] * (top - len(word_embeddings))
    words += ["<PAD>"] * (top - len(words))
    if len(word_embeddings) > top:
        word_embeddings = word_embeddings[:top]
        words = words[:top]
    return [words, word_embeddings]


train_df["input_embeddings"] = train_df.input_tokens.apply(lambda input_tokens: get_word_embeddings(input_tokens))
train_df[['input_tokens_pad', 'input_embeddings']] = pd.DataFrame(train_df.input_embeddings.tolist(),
                                                                  index=train_df.index)


def get_entity_embeddings(entities, ranked_output, top=20):
    entity_embeddings = []
    entity_list = []
    for entity in entities + ranked_output:
        if len(entity_embeddings) == top:
            break
        if entity not in entity_list and entity in entity_to_embedding:
            entity_list.append(entity)
            entity_embeddings.append(entity_to_embedding[entity])
    return entity_embeddings


train_df["output_embeddings"] = [get_entity_embeddings(output_entities, ranked_output) for
                                 output_entities, ranked_output in
                                 train_df[["output_entities", "ranked_output"]].values]

train_df["ranked_output"] = [[ent for ent in ranked_output if ent in entity_to_embedding] for ranked_output in train_df.ranked_output.values]


columns = ["input_tokens_pad", "input_embeddings", "output_embeddings", "ranked_output"]

train_df[columns].to_pickle("train_df_embeddings.pickle")
