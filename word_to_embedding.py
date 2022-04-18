import pickle

word_to_embedding = {}

with open("numberbatch-en.txt", "r", encoding="utf8") as fp:
    line = fp.readline()
    line = fp.readline()
    while line:
        try:
            line_tokens = line.split()
            word = line_tokens[0]
            emb = [float(num) for num in line_tokens[1:]]
            word_to_embedding[word.strip()] = emb
            line = fp.readline()
        except Exception as e:
            print(e)
with open("word_to_embedding.pickle", "wb") as fp:
    pickle.dump(word_to_embedding, fp)
    fp.close()
