import os
import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, LoraConfig, TaskType, get_peft_model, PeftModel
import numpy as np
from transformers import DataCollatorWithPadding, DataCollatorForLanguageModeling

# Ensure required packages are installed
# %pip install torch torchvision transformers datasets bitsandbytes peft

def configure_device():
    """Configure and return the device (GPU if available)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def initialize_distributed():
    """Initialize the distributed training environment."""
    print("Initializing the DDP environment")
    dist.init_process_group(backend='nccl')  # Use 'nccl' for multi-GPU training
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    print(f"Initialized distributed training on local rank: {local_rank}")
    return local_rank

def load_model(model_name: str):
    """Load and return the model and tokenizer from HuggingFace."""
    print("Starting to load the model")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def load_and_prepare_data(tokenizer, local_rank):
    """Load, process, and return the train and test datasets."""
    dataset = load_dataset("poornima9348/finance-alpaca-1k-test")
    train_test_split = dataset['test'].train_test_split(test_size=0.2, shuffle=True)
    train_dataset = train_test_split['train']
    test_dataset = train_test_split['test']

    # Drop unused columns
    train_dataset = train_dataset.remove_columns(['input', 'text'])
    test_dataset = test_dataset.remove_columns(['input', 'text'])

    # Define tokenization and prompt-building functions
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True)

    def prompt_builder(row):
        return {"text": row["instruction"] + row["output"]}

    # Apply the functions
    train_dataset = train_dataset.map(prompt_builder).map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(prompt_builder).map(tokenize_function, batched=True)

    # Convert datasets to PyTorch tensors
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    train_dataset = train_dataset.remove_columns(['instruction', 'output'])
    test_dataset = test_dataset.remove_columns(['instruction', 'output'])

    print("Setting up the dataset for distributed training")
    # Use DistributedSampler for DDP
    train_sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=local_rank)
    test_sampler = DistributedSampler(test_dataset, num_replicas=dist.get_world_size(), rank=local_rank)

    # Create DataLoader with DistributedSampler
    train_dataloader = DataLoader(train_dataset, batch_size=1, sampler=train_sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=1, sampler=test_sampler)
    return train_dataset, test_dataset

def prepare_lora_model(model):
    """Prepare the model for LoRA training."""
    peft_model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=32, 
        lora_alpha=32, 
        lora_dropout=0.1,
        task_type=TaskType.CAUSAL_LM,
    )
    lora_model = get_peft_model(peft_model, lora_config)
    return lora_model

def train_model(lora_model, train_dataset, test_dataset, tokenizer, local_rank):
    """Train the LoRA model with the specified datasets."""
    lora_model = DDP(lora_model.to(local_rank), device_ids=[local_rank])
    print("LoRA model is set up, starting the training now...")
    training_args = TrainingArguments(
        output_dir="Meta-Llama-3.1-8B-Instruct-finetuned",
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=1,
        weight_decay=0.01,
        #load_best_model_at_end=True,
        logging_steps=1,
        report_to="none",
        remove_unused_columns=False, 
        metric_for_best_model="eval_runtime"
    )

    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    trainer.train()
    print("Training completed!")
    return trainer

def generate_response(model, trained_model_path, tokenizer):
    """Generate and print a response using the fine-tuned model."""
    model_with_adapter = PeftModel.from_pretrained(model, trained_model_path).to("cuda")
    model_with_adapter.eval()
    inputs = tokenizer("How does a brokerage firm work?", return_tensors="pt")

    outputs = model_with_adapter.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=100)
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])

if __name__ == "__main__":
    # Configure device
    device = configure_device()
    
    # Initialize distributed training
    initialization = initialize_distributed()

    # Load model and tokenizer
    model_name = "NousResearch/Meta-Llama-3.1-8B-Instruct"
    model, tokenizer = load_model(model_name)

    # Load and prepare data
    #train_dataset, test_dataset = load_and_prepare_data(tokenizer)
    # Load and prepare data
    train_dataloader, test_dataloader = load_and_prepare_data(tokenizer, initialization)

    # Prepare the LoRA model
    lora_model = prepare_lora_model(model)

    # Train the model
    trainer = train_model(lora_model, train_dataloader, test_dataloader, tokenizer, initialization)

    # Generate a response with the trained model
    #generate_response(model=model, trained_model_path="Meta-Llama-3.1-8B-Instruct-finetuned/checkpoint-400", tokenizer=tokenizer)

    # Clean up distributed training
    dist.destroy_process_group()
