# Finetuning Llama 3.1 using LoRA

Fine-tune **Llama-3.1-8B** on the [Alpaca Financial Dataset](https://huggingface.co/datasets/poornima9348/finance-alpaca-1k-test) using **LoRA** (Low-Rank Adaptation) for parameter-efficient training.

## LoRA Overview

LoRA is a technique designed to make fine-tuning more efficient by focusing on reducing the number of trainable parameters involved. This approach speeds up the fine-tuning process and creates smaller fine-tuned checkpoints.

![LoRA Diagram](https://github.com/EkinKarabulut/finetuning_llama3.1/blob/main/lora_diagram.png?raw=true)

Rather than adjusting all the model's weights during fine-tuning, LoRA freezes most layers and selectively trains only a few within the attention mechanisms. Instead of directly modifying the weights of these layers, LoRA introduces two smaller matrices that are combined and added to the original weights. These smaller matrices are the only parts updated during fine-tuning and are saved separately. This method preserves the model's original parameters, allowing the LoRA weights to integrate later through an adaptation process seamlessly. Unloading the LoRA adapter and reverting back to the original base model is also possible.

For more information about LoRA, please refer to [the original paper](https://arxiv.org/abs/2106.09685).

---

## Hardware Requirements

| Component | Recommended |
|-----------|-------------|
| GPU | NVIDIA A100 / H100 / RTX 4090 (≥ 40 GB VRAM) |
| RAM | ≥ 32 GB |
| Storage | ≥ 50 GB (for model weights + dataset) |

For multi-GPU training, see the [Distributed Training](#distributed-training) section below.

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- CUDA 12.x
- A Hugging Face account with access to [Meta-Llama-3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B) (gated model — request access first)

### 2. Install dependencies

```bash
pip install torch torchvision transformers datasets bitsandbytes peft accelerate
```

### 3. Authenticate with Hugging Face

```bash
huggingface-cli login
```

### 4. Run the notebook

Open `Finetuning_Llama3.ipynb` in Jupyter:

```bash
jupyter notebook Finetuning_Llama3.ipynb
```

Or launch it directly in Google Colab / AWS SageMaker.

---

## Distributed Training

The `distributed/` directory contains scripts for multi-GPU training using `torchrun`:

```bash
cd distributed
bash launch.sh
```

To build and run with Docker:

```bash
cd distributed
docker build -t llama3-finetune .
docker run --gpus all llama3-finetune
```

---

## Repository Structure

```
finetuning_llama3.1/
├── Finetuning_Llama3.ipynb   # Main fine-tuning notebook
├── distributed/
│   ├── distributed.py        # Multi-GPU training script
│   ├── launch.sh             # torchrun launch script
│   └── Dockerfile            # Docker image for distributed training
├── lora_diagram.png          # LoRA architecture diagram
└── README.md
```

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora_r` | 16 | LoRA rank (controls adapter size) |
| `lora_alpha` | 32 | LoRA scaling factor |
| `lora_dropout` | 0.05 | Dropout probability for LoRA layers |
| `num_train_epochs` | 3 | Number of training epochs |
| `per_device_train_batch_size` | 4 | Batch size per GPU |
| `learning_rate` | 2e-4 | Learning rate |

---

## License

This project builds on Meta's Llama 3.1 model. Usage is subject to the [Meta Llama 3 Community License](https://llama.meta.com/llama3/license/).
