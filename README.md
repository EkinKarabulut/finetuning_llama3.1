# Finetuning Llama3.1 using LoRA 
In this notebook, we will finetune Llama-3.1-8B with [Alpaca Financial Dataset](https://huggingface.co/datasets/poornima9348/finance-alpaca-1k-test). 

## LoRA

LoRA is a technique designed to make fine-tuning more efficient by focusing on reducing the number of trainable parameters involved. This approach speeds up the fine-tuning process and creates smaller fine-tuned checkpoints.

![LoRA](https://github.com/EkinKarabulut/finetuning_llama3.1/blob/main/lora_diagram.png?raw=true)

Rather than adjusting all the model's weights during fine-tuning, LoRA freezes most layers and selectively trains only a few within the attention mechanisms. Instead of directly modifying the weights of these layers, LoRA introduces two smaller matrices that are combined and added to the original weights. These smaller matrices are the only parts updated during fine-tuning and are saved separately. This method preserves the model's original parameters, allowing the LoRA weights to integrate later through an adaptation process seamlessly. Unloading the LoRA adapter and revert back to the original base model is also possible. 

For more information about LoRA, please refer to [this paper](https://arxiv.org/abs/2106.09685).
