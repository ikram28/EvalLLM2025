# ========== IMPORTS ==========
import os
import sys
import json
import torch
import logging
import argparse
import datasets
from datasets import Dataset, load_dataset
import transformers
from transformers import set_seed, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig
import wandb
import torch.distributed as dist
import idr_torch

# ========== WANDB CONFIGURATION ==========
os.environ['WANDB_MODE'] = 'offline'
wandb_project = "SFT-NER-LLAMA"  
if wandb_project:
    os.environ["WANDB_PROJECT"] = wandb_project

logger = logging.getLogger(__name__)

# ========== MAIN FUNCTION ==========
def main():
    # ----------- Argument Parsing -----------
    parser = argparse.ArgumentParser(
        description="Supervised fine-tuning for NER with Llama and LoRA.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model_name", type=str, required=True,
                        help="HuggingFace Hub model name (e.g., 'meta-llama/Llama-2-7b-hf').")
    parser.add_argument("--path_train_dataset", type=str, default="./data/train-XML.json",
                        help="Path to train dataset (JSON list of dicts).")
    parser.add_argument("--path_eval_dataset", type=str, default="./data/validation-XML.json",
                        help="Path to eval dataset (JSON list of dicts).")
    parser.add_argument("--output_dir", type=str, default="./output-models/",
                        help="Directory to save model checkpoints.")
    parser.add_argument("--logging_dir", type=str, default="./output-logs/",
                        help="Directory to save logs.")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size per device during training.")
    parser.add_argument("--save_steps", type=int, default=10,
                        help="Steps between model checkpoint saves.")
    parser.add_argument("--logging_steps", type=int, default=5,
                        help="Steps between logging events.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")

    args = parser.parse_args()

    # ----------- TrainingArguments -----------
    training_args = transformers.TrainingArguments(
        do_eval=True,
        eval_strategy="steps",
        eval_steps=10,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        logging_strategy="steps",
        lr_scheduler_type="cosine",
        num_train_epochs=args.epochs,
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        push_to_hub=False,
        remove_unused_columns=True,
        report_to="wandb",
        save_strategy="steps",
        save_steps=args.save_steps,
        seed=args.seed,
        logging_dir=args.logging_dir,
        logging_first_step=True,
        optim="adamw_torch",
        ddp_find_unused_parameters=False,
        local_rank=idr_torch.local_rank,
    )
    set_seed(args.seed)

    # ----------- Logging Setup -----------
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu} "
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # ----------- Tokenizer -----------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, model_max_length=2048, padding=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'

    # ----------- Prompt Template (System Prompt) -----------
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
Use the exact label name as the tag.
"""
)

    # ----------- Formatting Function -----------
    def formatting_prompts_func(instances):
        messages = []
        for instance in instances:
            chat = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Text:\n{instance['text']}"},
                {"role": "assistant", "content": f"Annotations:\n{instance['tagged_text']}"}
            ]
            messages.append({"chat": chat})
        return messages

    # ----------- Load Datasets -----------
    with open(args.path_train_dataset, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(args.path_eval_dataset, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)

    train_dataset = formatting_prompts_func(train_data)
    eval_dataset = formatting_prompts_func(eval_data)
    train_dataset = Dataset.from_list(train_dataset)
    eval_dataset = Dataset.from_list(eval_dataset)

    # ----------- Apply Chat Template -----------
    train_dataset = train_dataset.map(
        lambda x: {"formatted_chat": tokenizer.apply_chat_template(x["chat"], tokenize=False, add_generation_prompt=False)}
    )
    eval_dataset = eval_dataset.map(
        lambda x: {"formatted_chat": tokenizer.apply_chat_template(x["chat"], tokenize=False, add_generation_prompt=False)}
    )

    # ----------- Model Loading -----------
    logger.info("*** Load pretrained model ***")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model_kwargs = dict(
        trust_remote_code=True,
        use_flash_attention_2=True,
        torch_dtype=torch.float16,
        use_cache=not training_args.gradient_checkpointing,
        quantization_config=quantization_config,
    )

    # ----------- PEFT/LoRA Configuration -----------
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        modules_to_save=None,
    )


    # ----------- Trainer Initialization -----------
    trainer = SFTTrainer(
        model=args.model_name,
        model_init_kwargs=model_kwargs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        max_seq_length=2048,
        tokenizer=tokenizer,
        dataset_text_field="formatted_chat",
        peft_config=peft_config,
    )

    # ----------- Training Loop -----------
    logger.info("***** Starting Training *****")
    train_result = trainer.train()

    # ----------- Save Model & Metrics -----------
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_model(args.output_dir)
    trainer.tokenizer.save_pretrained(args.output_dir)
    trainer.model.config.use_cache = True
    trainer.model.config.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()

