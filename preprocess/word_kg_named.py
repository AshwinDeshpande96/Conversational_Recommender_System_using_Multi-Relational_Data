import pickle

import pandas as pd
from tqdm import tqdm

data = {}
keys = ["word_kg_clean", "word_to_embedding", "id_to_word", "id_to_word_relation"]
for key in keys:
    with open(f"final_data/{key}.pickle", "rb") as fp:
        data[key] = pickle.load(fp)
        fp.close()




def main():
    ############## edge
    edge_columns = ["source", "target", "relation"]
    edge_df = pd.DataFrame(columns=edge_columns)
    word_to_embedding = {}
    for w1id, rid, w2id in tqdm(data["word_kg_clean"]):
        w1 = data["id_to_word"][w1id]
        r = data["id_to_word_relation"][rid]
        w2 = data["id_to_word"][w2id]
        row = [w1, w2, r]
        if w1 not in word_to_embedding:
            word_to_embedding[w1] = data["word_to_embedding"][w1]
        if w2 not in word_to_embedding:
            word_to_embedding[w2] = data["word_to_embedding"][w2]
        edge_df.loc[len(edge_df.index)] = row
    edge_df = edge_df[~edge_df[["source", "target"]].duplicated()]

    ############# node
    wcolumns = ["node"] + [f"f{i + 1}" for i in range(300)]
    node_df = pd.DataFrame(columns=wcolumns)
    for word in tqdm(word_to_embedding):
        embedding = word_to_embedding[word]
        row = [word] + list(embedding)
        node_df.loc[len(node_df.index)] = row
    node_df = node_df.set_index("node")
    edge_df.to_pickle("word_edge_df.pickle")
    node_df.to_pickle("word_node_df.pickle")
    print()


if __name__ == '__main__':
    main()
