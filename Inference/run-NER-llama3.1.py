# ========== IMPORTS ==========
import os
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import huggingface_hub

# ========== ENVIRONMENT AND DEVICE SETUP ==========
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ========== HF TOKEN LOGIN ==========
with open("HF-TOKEN.txt", "r") as f:
    hf_token = f.read().strip()
huggingface_hub.login(hf_token)

# ========== CONFIGURATION ==========
EMBED_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'
K = 10  # Number of few-shot examples
SYSTEM_PROMPT = ("""Extract entities from the following French health-related text by evaluating key phrases and determining if they are entities according to the following annotation guide.
* Annotation Rules:
- Don't include articles and determiners (e.g., annotate "coqueluche" not "la coqueluche")
- Punctuation marks (comma, period, etc.) should not be included, except when they are part of the entity’s name (e.g., "Brucella spp.", "Bosch Inc.").
- Annotate acronyms separately (e.g., "OMS" in "Organisation Mondiale de la Santé (OMS)").
- Annotations must not overlap or cross sentence boundaries except for discontinuous entities, meaning entities that their parts are separated in the text (e.g., "Agence régionale de santé (ARS) d’Île de France" or "virus de la dengue et du chikungunya"). In such cases, the entire entity, including all relevant parts, should be captured and annotated with its appropriate label (e.g., "ORGANIZATION" or "PATHOGEN").
- Don't annotate misspelled words unless they're transliterations
- Don't annotate generic terms or pronouns referring to entities
- Only use one of the 21 labels below.  
- If you think a span is an entity but it doesn’t fit one of those labels, **do not tag** it.  
- Never invent or use any other tag. 
* Entity Types:
**Authors and Sources:
- DOC_AUTHOR: designates the document's author(s).
- DOC_SOURCE: designates the document's source(s). For example: 'Agence France Presse', 'Reuters,' 'Le Midi Libre', etc.
** Diseases and Pathogens:
- INF_DISEASE:  Diseases in humans, animals, or plants that are caused by specific infectious agents.
- NON_INF_DISEASE:  Diseases in humans, animals, or plants that are not caused by infectious agents.
- PATHOGEN:  a bacterium, virus, parasite, prion, or fungus that can cause an infectious disease. Generic terms such as ‘bactérie’, 'virus', 'parasite', 'prion', 'levure', 'champignon', etc. without any precision are not annotated."
- DIS_REF_TO_PATH:the name of an infectious disease used to refer to a pathogen. For example, in the sentence 'de nombreux parasites tels que le paludisme' the term 'paludisme' refers to the Plasmodium spp. parasites and should be annotated as 'DIS_REF_TO_PATH'.
- PATH_REF_TO_DIS:the name of a pathogen used to designate an infectious disease. For example, in the sentence 'le nombre de cas de VIH est en augmentation', the term 'VIH' is used to refer to the disease (HIV infection) and should be annotated as 'PATH_REF_TO_DIS'.
- NRBCE Agents and Toxins:
- RADIOISOTOPE: an unstable form of an element that emits radiation (e.g., césium-137). Element families that consist solely of unstable isotopes should also be annotated, even if the atomic number of the element is not specified (e.g., uranium, polonium, etc.)
- TOXIC_C_AGENT: ny inorganic chemical agent that is toxic to humans, animals, or plants.
- EXPLOSIVE: Explosive substances.
- BIO_TOXIN: any organic chemical substance that is toxic to one or more living species.
** Locations and Organizations:
- LOCATION: a named geographic area (continent, pays, région, ville, quartier, rue, montagne, rivière, mer,  etc.). These areas lack any inherent intent. If a voluntary action or statement can be linked to a place, that place should be annotated as an organization (see LOC_REF_TO_ORG).
- ORGANIZATION:  a legal or social entity – excluding countries, regions, departments, prefectures, municipalities, etc. – that can be identified without context (e.g. institution, entreprise, agence, organisation non gouvernementale, parti politique, armée, gouvernement, etc.). For example, the sequences 'Tribunal judiciaire de Paris,' 'hôpital Georges Pompidou' and 'gouvernement français' should be annotated as ORGANIZATION.
- LOC_REF_TO_ORG: an organization designated solely by a place name. For example, in the sentence 'Paris a décidé de faire construire un nouveau métro',  'Paris' is annotated as LOC_REF_TO_ORG because it could be replaced by 'la mairie de Paris’.
- ORG_REF_TO_LOC: an organization’s name that is used to designate the geographic area where it is located. In the text: 'des fumées toxiques s’échappent de la centrale nucléaire de Tchernobyl',  the segment 'centrale nucléaire de Tchernobyl' is annotated as ORG_REF_TO_LOC.
** Dates:
- ABS_DATE: Absolute dates (e.g., "15 mars 2020").
- REL_DATE: Relative dates (e.g., "hier", “lundi dernier”, “8 janvier”).
- DOC_DATE: Document publication date. It can be absolute or relative.
- ABS_PERIOD: Absolute periods (e.g., "mars 2020", “ du 1er au 3 mai 2024”, “semaine 51 de 2020”, “20ème siècle”).
- REL_PERIOD: Relative periods (e.g., “mars prochain”, les “3 derniers jours”, “du 10 au 20 mai”, la "semaine dernière").
- FUZZY_PERIOD: A vague time expression describing a period longer than a day without precise limits, needing additional context for proper interpretation (e.g., "depuis plusieurs semaines",  à la “fin de la semaine”, au “début du mois”, “ces dernières années”, “depuis 3 ou 4 mois”).
* Output Format:
Return _only_ the original text augmented with XML tags around each entity mention.
Use the exact label name as the tag. Here is an example to help you understand your task better.
"""
)

# ========== LOAD DATA ==========
with open('data/test.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)
test_texts = [d['text'] for d in test_data]

with open('data/train_fewshot.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)
train_texts = [d['text'] for d in train_data]
train_tagged = [d['tagged_text'] for d in train_data]

# ========== COMPUTE EMBEDDINGS ==========
embedder = SentenceTransformer(EMBED_MODEL)
test_embeddings = embedder.encode(test_texts, convert_to_numpy=True)
train_embeddings = embedder.encode(train_texts, convert_to_numpy=True)

# ========== FEW-SHOT SELECTION BY SIMILARITY ==========
def top_k_similar(test_emb, train_embs, k=10):
    sims = cosine_similarity([test_emb], train_embs)[0]
    return np.argsort(sims)[-k:][::-1]

# ========== LLM & TOKENIZER LOADING ==========
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)
MODEL_NAME = "ik-ram28/NER-LLama-3.1-8B"  
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16
)

# ========== RUN INFERENCE ==========
results = []
with torch.no_grad():
    for i, (text, emb) in enumerate(zip(test_texts, test_embeddings)):
        # Select most similar few-shot examples
        neighbors = top_k_similar(emb, train_embeddings, k=K)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        # Add few-shot examples as chat turns
        for nb in neighbors:
            messages.append({"role": "user", "content": f"Text:\n{train_texts[nb]}"})
            messages.append({"role": "assistant", "content": f"Annotations:\n{train_tagged[nb]}"})
        # Add the actual task
        messages.append({"role": "user", "content": f"Text:\n{text}\nAnnotations:"})

        # Format with chat template
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        input_ids = inputs.to(device)

        outputs = model.generate(
            input_ids,
            max_new_tokens=2048,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True
        )
        generated_ids = outputs.sequences

        # Remove the prompt tokens from output
        new_generated_ids = generated_ids[0][input_ids.shape[1]:]
        new_generated_text = tokenizer.decode(new_generated_ids, skip_special_tokens=True)

        results.append({
            "text": text,
            "prediction": new_generated_text,
        })

# ========== SAVE PREDICTIONS ==========
with open("output/ner_results_llama.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
