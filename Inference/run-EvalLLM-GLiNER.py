# ========== IMPORTS ==========
from gliner import GLiNER
import json
import torch
from tqdm import tqdm

# ========== DEVICE SETUP ==========
device = "cuda" if torch.cuda.is_available() else "cpu"

# ========== MODEL LOADING ==========
MODEL = "ik-ram28/EvalLLM-GLiNER"  
model = GLiNER.from_pretrained(MODEL).to(device)

# ========== LABELS ==========
labels = [
    "Maladie infectieuse",
    "Maladie non infectieuse",
    "Agent pathogène",
    "Nom de maladie désignant un pathogène",
    "Nom de pathogène désignant une maladie",
    "Isotope radioactif",
    "Agent chimique toxique",
    "Substance explosive",
    "Toxine biologique",
    "Location",
    "Organisation",
    "Lieu désignant une organisation",
    "Organisation désignant un lieu",
    "Date absolue",
    "Date relative",
    "Date de publication du document",
    "Période absolue",
    "Période relative",
    "Période floue",
    "Source du document",
    "Auteur du document"
]

# ========== LOAD INPUT DOCUMENTS ==========
INPUT_FILE = "data/test_documents.json"  
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    documents = json.load(f)

# ========== INFERENCE ==========
results = []
for doc in tqdm(documents):
    doc_id = doc.get("id", None)
    text = doc["text"]
    preds = model.predict_entities(text, labels, threshold=0.4)
    results.append({
        "id": doc_id,
        "text": text,
        "entities": preds
    })

# ========== SAVE OUTPUT ==========
OUTPUT_FILE = "output/gliner_predictions.json"  
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Prediction complete. Results saved to {OUTPUT_FILE}")
