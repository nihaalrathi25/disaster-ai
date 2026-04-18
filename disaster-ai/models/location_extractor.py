
import spacy
import re

nlp = spacy.load("en_core_web_sm")

def extract_location(tweet, bio=""):

    # 1. hashtag first
    hashtags = re.findall(r"#(\w+)", tweet)
    if hashtags:
        return hashtags[0]

    # 2. NER
    text = str(tweet) + " " + str(bio)
    doc = nlp(text)

    for ent in doc.ents:
        if ent.label_ == "GPE":
            return ent.text

    return "Unknown"
