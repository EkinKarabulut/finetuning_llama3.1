FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN pip install torch torchvision transformers datasets bitsandbytes peft

# Set the working directory 
WORKDIR /model

# Copy the current directory contents into the container at /model
COPY distributed.py launch.sh /model/

# Create the checkpoint directory
RUN mkdir -p /model/checkpoints && chmod -R 777 /model/checkpoints

# Run the bash file
RUN chmod u+x launch.sh
CMD ["./launch.sh"]
