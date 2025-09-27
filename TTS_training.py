from datasets import load_dataset, Audio
from transformers import (
    CsmForConditionalGeneration,
    TrainingArguments,
    CsmProcessor,
    Trainer
)
import torch
# Carica processor e modello pre-addestrato
processor = CsmProcessor.from_pretrained("sesame/csm-1b")
model = CsmForConditionalGeneration.from_pretrained("sesame/csm-1b")
model.train()
model.codec_model.eval()

# Carica il dataset Common Voice italiano
ds = load_dataset("mozilla-foundation/common_voice_11_0", "it", split="train")
ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

# Data collator prepara coppie audio-testo
def data_collator(samples):
    conversations = []

    for sample in samples:
        # Nel Common Voice ogni elemento è un singolo utterance, quindi struttura con un solo step conversazionale
        conversation = [{
            "role": str(sum(ord(c) for c in sample["client_id"])),
            "content": [
                {"type": "text", "text": sample["sentence"]},
                {"type": "audio", "audio": sample["audio"]["array"]}
            ]
        }]
        conversations.append(conversation)

    inputs = processor.apply_chat_template(
        conversations,
        tokenize=True,
        return_dict=True,
        output_labels=True,
    )
    return inputs


# Configurazione del training
training_args = TrainingArguments(
    output_dir="commonvoice-it-trainer",
    remove_unused_columns=False,
    per_device_train_batch_size=1,
    gradient_checkpointing=True,
    fp16=True,
    gradient_accumulation_steps=32,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    data_collator=data_collator,
)

# Avvia il training
torch.cuda.empty_cache()

trainer.train()
model.save_pretrained("commonvoice-it-finetuned-model")
processor.save_pretrained("commonvoice-it-finetuned-processor")