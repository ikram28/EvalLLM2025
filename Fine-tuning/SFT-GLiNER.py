# ========== IMPORTS ==========
import os
import json
import torch
import wandb
from gliner import GLiNER
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import DataCollator
from gliner.data_processing import GLiNERDataset

# ========== WANDB AND ENVIRONMENT SETUP ==========
os.environ["TOKENIZERS_PARALLELISM"] = "true"

with open("wandb.txt", "r") as f:
    wandb_api_key = f.read().strip()
wandb.login(key=wandb_api_key)

wandb_project = "FT-GLiNER"
if wandb_project:
    os.environ["WANDB_PROJECT"] = wandb_project

# ========== DATA PATHS ==========
train_path = "data/train_gliner.json"
val_path = "data/validation_gliner.json"

# ========== LOAD DATA ==========
with open(train_path, "r") as f:
    train_data = json.load(f)

with open(val_path, "r") as f:
    val_data = json.load(f)

# ========== MODEL AND DEVICE ==========
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GLiNER.from_pretrained("Ihor/gliner-biomed-large-v1.0").to(device)


# ========== DATA COLLATOR ==========
data_collator = DataCollator(
    model.config,
    data_processor=model.data_processor,
    prepare_labels=True
)

# ========== TRAINING PARAMETERS ==========
num_steps = 600
batch_size = 8
data_size = len(train_data)
num_batches = data_size // batch_size
num_epochs = max(1, num_steps // num_batches)

training_args = TrainingArguments(
    output_dir="new-models",                   
    learning_rate=1e-5,
    weight_decay=0.01,
    others_lr=5e-5,
    others_weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    eval_strategy="steps",
    eval_steps=50,
    save_steps=50,
    dataloader_num_workers=0,
    use_cpu=(device == "cpu"),
    report_to="wandb",
)

# ========== TRAINER INITIALIZATION ==========
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=model.data_processor.transformer_tokenizer,
    data_collator=data_collator,
)

# ========== TRAIN MODEL ==========
trainer.train()
