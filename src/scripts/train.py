# scripts/train.py - Qwen TSQL Text-to-SQL Fine-Tuning with DeepSpeed Zero-3 (3x 24GB GPUs)

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "Qwen/Qwen3.5-27B"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto",
)

# Load ChatML dataset from JSONL
dataset = load_dataset(
    "json",
    data_files="../data/synthetic_instruct/tsql_instruct_chatml.jsonl",
    split="train"
)


def format_chatml(examples):
    """Format examples as ChatML strings for training."""
    formatted = []
    for messages in examples["messages"]:
        chatml = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                chatml += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                chatml += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        chatml += "<|im_start|>assistant\n"
        formatted.append(chatml)
    return {"text": formatted}


# Format dataset as ChatML
dataset = dataset.map(
    format_chatml,
    batched=True,
    remove_columns=["messages"],
)

# DeepSpeed Zero-3 config for 3x 24GB GPUs
deepspeed_config = {
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": False,
        "offload_param": False,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e7,
    },
    "fp16": {
        "enabled": True,
        "auto_scale": True,
    },
    "train_batch_size": 12,  # 4 per GPU x 3 GPUs
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,
}

# Training arguments
# max_seq_length is 1536 (lower than C# trainer's 2048 — TSQL units are shorter)
training_args = SFTConfig(
    output_dir="../outputs/qwen-tsql-specialized",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,  # 4 x 3 GPUs = 12 total batch size
    gradient_accumulation_steps=4,
    fp16=True,
    save_steps=1000,
    save_total_limit=2,
    logging_steps=100,
    warmup_steps=100,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    learning_rate=5e-5,
    max_seq_length=1536,
    dataset_text_field="text",
    deepspeed=deepspeed_config,
)

# Trainer setup
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

# Train
trainer.train()

# Save model
trainer.save_model("../outputs/qwen-tsql-specialized")
