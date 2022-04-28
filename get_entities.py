import json
import pickle
import re
import time
from pprint import pprint

import neuralcoref
import requests
import spacy
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

porter = PorterStemmer()
stop_words = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_sm')
neuralcoref.add_to_pipe(nlp)
contractions = {
    "ain't": "am not / are not / is not / has not / have not",
    "aren't": "are not / am not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had / he would",
    "he'd've": "he would have",
    "he'll": "he shall / he will",
    "he'll've": "he shall have / he will have",
    "he's": "he has / he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how has / how is / how does",
    "i'd": "i had / i would",
    "i'd've": "i would have",
    "i'll": "i shall / i will",
    "i'll've": "i shall have / i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it had / it would",
    "it'd've": "it would have",
    "it'll": "it shall / it will",
    "it'll've": "it shall have / it will have",
    "it's": "it has / it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had / she would",
    "she'd've": "she would have",
    "she'll": "she shall / she will",
    "she'll've": "she shall have / she will have",
    "she's": "she has / she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as / so is",
    "that'd": "that would / that had",
    "that'd've": "that would have",
    "that's": "that has / that is",
    "there'd": "there had / there would",
    "there'd've": "there would have",
    "there's": "there has / there is",
    "they'd": "they had / they would",
    "they'd've": "they would have",
    "they'll": "they shall / they will",
    "they'll've": "they shall have / they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had / we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what shall / what will",
    "what'll've": "what shall have / what will have",
    "what're": "what are",
    "what's": "what has / what is",
    "what've": "what have",
    "when's": "when has / when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where has / where is",
    "where've": "where have",
    "who'll": "who shall / who will",
    "who'll've": "who shall have / who will have",
    "who's": "who has / who is",
    "who've": "who have",
    "why's": "why has / why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had / you would",
    "you'd've": "you would have",
    "you'll": "you shall / you will",
    "you'll've": "you shall have / you will have",
    "you're": "you are",
    "you've": "you have"
}


def get_alpha(text):
    return re.sub("[^a-z]", "", text)


contractions.update({get_alpha(key): value for key, value in contractions.items()})
try:
    with open(r"word_to_root.pickle", "rb") as vocab_file:
        word_to_root = pickle.load(vocab_file)
        vocab_file.close()
except:
    word_to_root = {}
try:
    with open(r"words.pickle", "rb") as vocab_file:
        words = pickle.load(vocab_file)
        vocab_file.close()
except:
    words = {}
try:
    with open(r"word_kg.pickle", "rb") as wkg_file:
        word_kg = pickle.load(wkg_file)
        wkg_file.close()
except:
    word_kg = []
try:
    with open(r"train_data.pickle", "rb") as data_file:
        result_list = pickle.load(data_file)
        data_file.close()
except:
    result_list = {}

with open('website/train_data.jsonl', 'r') as json_file:
    json_list = list(json_file)
    json_file.close()


def preprocess(messages):
    prev_sender = None
    conversation = []
    for message in messages:
        if message['senderWorkerId'] != prev_sender:
            conversation.append({"text": message['text'],
                                 "senderWorkerId": message["senderWorkerId"]})
            prev_sender = message['senderWorkerId']
        else:
            conversation[-1]['text'] += ". " + message["text"]
    conversation = get_entities(conversation)
    # TODO: Join previous n respondent utterances as context
    return conversation


def extract_entities_dbpedia(text, confidence=0.75):
    # doc = nlp(text)
    # time.sleep(1)
    entities = []
    tokens = re.split("\s+", text)
    text = text.lower()
    argument = "%20".join(tokens)
    try:
        obj = requests.get(
            f"https://api.dbpedia-spotlight.org/en/annotate?text={argument}&confidence={confidence}",
            headers={"Accept": "application/json"})
        data = json.loads(obj.content)
        for resource in data.get("Resources", []):
            uri = resource["@URI"]
            ent = resource["@surfaceForm"].lower()
            text = text.replace(ent, " ")
            entities.append(uri)
    except Exception as e:
        print(e)
    text = re.sub(r"\s+", " ", text).strip()
    return text, entities


def extract_entities_spacy(text):
    doc = nlp(text)
    entities = []
    i = 0
    val = [(X, X.ent_iob_, X.ent_type_) for X in doc]
    while i < len(doc):
        X = doc[i]
        if X.ent_iob_ == "O":
            i += 1
            continue
        if re.match(r"\@[0-9]+", X.string):
            i += 1
            continue
        if X.ent_iob_ == "U":
            entities.append(X.string.strip())
            i += 1
            continue
        if X.ent_iob_ == "B":
            j = i + 1
            while j < len(doc) and doc[j].ent_iob_ in ["L", "I"]:
                j += 1
            ent = " ".join([x.string for x in doc[i:j]]).strip()
            ent = re.sub("\s+", " ", ent).strip()
            entities.append(ent)
            i = j + 1
    for ent in entities:
        text = text.replace(ent, " ")
    text = re.sub("\s+", " ", text).strip()
    return text, entities


def extract_movies(text):
    movies = re.findall(r"\@[0-9]+", text)
    return movies


def fetch_conceptnet(word):
    time.sleep(1)
    obj = requests.get(f"https://api.conceptnet.io/query?node=/c/en/{word}&other=/c/en&limit=50")
    print(word, obj.status_code)
    word_data = json.loads(obj.content)
    wid = f"/c/en/{word}"
    for edge in word_data["edges"]:
        w1 = edge["start"]["@id"]
        w2 = edge["end"]["@id"]
        if wid != w1 and wid != w2:
            continue
        r = edge['rel']['@id']
        if (w1, r, w2) not in word_kg:
            word_kg.append((w1, r, w2))


def clean_text(text):
    text = re.sub(r"[^a-z ]", " ", text.lower())
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if w not in stop_words]
    i = 0
    while i < len(filtered_sentence):
        word = filtered_sentence[i]
        i += 1
        word_root = porter.stem(word)
        if word in ["nd", "th", "rd"]:
            filtered_sentence.remove(word)
            i -= 1
            continue
        word_to_root[word] = word_root
        if word_root in words:
            continue
        words[word_root] = word
        fetch_conceptnet(word)
    return filtered_sentence


def get_entities(messages):
    for i, message in enumerate(messages):
        text = re.sub("[^a-zA-Z' ]", " ", message["text"]).strip()
        tokens = re.split("\s+", text)
        text = " ".join([contractions[w.lower()] if w.lower() in contractions else w for w in tokens])
        doc = nlp(text)
        resolved_text = doc._.coref_resolved
        messages[i]["text"] = resolved_text
        text_wo_ent, messages[i]["entities"] = extract_entities_dbpedia(resolved_text)
        messages[i]["movies"] = extract_movies(resolved_text)
        messages[i]["tokens"] = clean_text(text_wo_ent)
    return messages


for json_str in json_list:
    result = json.loads(json_str)
    # if result['conversationId'] in result_list:
    #     continue
    result["conversation"] = preprocess(result['messages'])
    pprint(result)
    result_list[result['conversationId']] = result
    # with open(r"train_data.pickle", "wb") as train_file:
    #     pickle.dump(result_list, train_file)
    #     train_file.close()
    #
    # with open(r"words.pickle", "wb") as vocab_file:
    #     pickle.dump(words, vocab_file)
    #     vocab_file.close()
    #
    # with open(r"word_to_root.pickle", "wb") as vocab_file:
    #     pickle.dump(word_to_root, vocab_file)
    #     vocab_file.close()
    #
    # with open(r"word_kg.pickle", "wb") as kg_file:
    #     pickle.dump(word_kg, kg_file)
    #     kg_file.close()
