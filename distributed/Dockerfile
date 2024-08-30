FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN pip install torch torchvision transformers datasets bitsandbytes peft

# Set the working directory 
WORKDIR /demo

# Copy the current directory contents into the container at /demo
COPY distributed.py launch.sh /demo/

# Set environment variable for checkpoint directory
#ENV CHECKPOINT_DIR=/demo/Meta-Llama-3.1-8B-Instruct-finetuned/checkpoint-400

# Create the checkpoint directory
#RUN mkdir -p $CHECKPOINT_DIR && chmod -R 777 /demo/Meta-Llama-3.1-8B-Instruct-finetuned

RUN mkdir -p /demo/Meta-Llama-3.1-8B-Instruct-finetuned && chmod -R 777 /demo/Meta-Llama-3.1-8B-Instruct-finetuned

# Run the bash file
RUN chmod u+x launch.sh
CMD ["./launch.sh"]
