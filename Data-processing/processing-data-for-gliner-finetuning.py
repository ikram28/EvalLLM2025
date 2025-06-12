# ========== IMPORTS ==========
import json
import os
import re
import random

# ========== LABEL MAPPINGS ==========
label_map = {
    "Maladie infectieuse": "INF_DISEASE",
    "Maladie non infectieuse": "NON_INF_DISEASE",
    "Agent pathogène": "PATHOGEN",
    "Nom de maladie désignant un pathogène": "DIS_REF_TO_PATH",
    "Nom de pathogène désignant une maladie": "PATH_REF_TO_DIS",
    "Isotope radioactif": "RADIOISOTOPE",
    "Agent chimique toxique": "TOXIC_C_AGENT",
    "Substance explosive": "EXPLOSIVE",
    "Toxine biologique": "BIO_TOXIN",
    "Lieu géographique": "LOCATION",
    "Organisation": "ORGANIZATION",
    "Lieu désignant une organisation": "LOC_REF_TO_ORG",
    "Organisation désignant un lieu": "ORG_REF_TO_LOC",
    "Date absolue": "ABS_DATE",
    "Date relative": "REL_DATE",
    "Date de publication du document": "DOC_DATE",
    "Période absolue": "ABS_PERIOD",
    "Période relative": "REL_PERIOD",
    "Période floue": "FUZZY_PERIOD",
    "Source du document": "DOC_SOURCE",
    "Auteur du document": "DOC_AUTHOR"
}
inverse_label_map = {v: k for k, v in label_map.items()}

# ========== TOKENIZER ==========
def tokenize_text(text):
    """Tokenizes the input text into a list of tokens."""
    return re.findall(r'\w+(?:[-_]\w+)*|\S', text)

# ========== CONVERSION FUNCTION ==========
def convert_to_gliner_format(documents):
    gliner_data = []
    for doc in documents:
        text = doc['text']
        entities = doc['entities']

        tokens = tokenize_text(text)
        spans = []
        start = 0
        for token in tokens:
            match = re.search(re.escape(token), text[start:])
            if match:
                token_start = start + match.start()
                token_end = token_start + len(token)
                spans.append((token_start, token_end))
                start = token_end
            else:
                spans.append((None, None))

        ner = []
        for entity in entities:
            try:
                char_start = entity["start"][0] if isinstance(entity["start"], list) else entity["start"]
                char_end = entity["end"][0] if isinstance(entity["end"], list) else entity["end"]
                label_code = entity["label"]
                label = inverse_label_map.get(label_code, label_code)

                token_start_idx = token_end_idx = None
                for idx, (tok_start, tok_end) in enumerate(spans):
                    if tok_start is not None and tok_end is not None:
                        if token_start_idx is None and tok_start <= char_start < tok_end:
                            token_start_idx = idx
                        if tok_start < char_end <= tok_end or (char_end == tok_end):
                            token_end_idx = idx
                            break
                if token_start_idx is not None and token_end_idx is not None:
                    ner.append([token_start_idx, token_end_idx, label])
                else:
                    print(f"⚠️ Could not map entity: {entity} in text: {text}")
            except (KeyError, IndexError, TypeError) as e:
                print(f"Skipping invalid entity: {entity}, error: {e}")

        gliner_doc = {
            "tokenized_text": tokens,
            "ner": ner
        }
        gliner_data.append(gliner_doc)
    return gliner_data


# ========== LOAD RAW DATA ==========
INPUT_FILE = "data/raw_evalllm.json"   
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)
print('Dataset size:', len(data))

# ========== CONVERT TO GLINER FORMAT ==========
data = convert_to_gliner_format(data)
with open("preprocessed_gliner_data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

# ========== SPLIT INTO TRAIN/VALID ==========
random.seed(42)
random.shuffle(data)
print('Dataset is shuffled...')

split_point = int(len(data) * 0.8)
train_docs = data[:split_point]
validation_docs = data[split_point:]

# ========== SAVE TRAIN/VALID SPLITS ==========
with open("train_gliner.json", "w", encoding="utf-8") as f:
    json.dump(train_docs, f, indent=4, ensure_ascii=False)
with open("validation_gliner.json", "w", encoding="utf-8") as f:
    json.dump(validation_docs, f, indent=4, ensure_ascii=False)

print("Conversion and splitting done. Files written: preprocessed_gliner_data.json, train_gliner.json, validation_gliner.json")
