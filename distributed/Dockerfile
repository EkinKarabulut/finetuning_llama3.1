FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN pip install torch torchvision transformers datasets bitsandbytes peft

# Set the working directory 
WORKDIR /demo

# Copy the current directory contents into the container at /demo
COPY distributed.py launch.sh /demo/

# Run the bash file
RUN chmod u+x launch.sh
CMD ["./launch.sh"]
