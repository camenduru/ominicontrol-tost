FROM ubuntu:22.04

WORKDIR /content

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=True
ENV PATH="/home/camenduru/.local/bin:/usr/local/cuda/bin:${PATH}"

RUN apt update -y && apt install -y software-properties-common build-essential \
    libgl1 libglib2.0-0 zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev && \
    add-apt-repository -y ppa:git-core/ppa && apt update -y && \
    apt install -y python-is-python3 python3-pip sudo nano aria2 curl wget git git-lfs unzip unrar ffmpeg && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda_12.6.2_560.35.03_linux.run -d /content -o cuda_12.6.2_560.35.03_linux.run && sh cuda_12.6.2_560.35.03_linux.run --silent --toolkit && \
    echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf && ldconfig && \
    git clone https://github.com/aristocratos/btop /content/btop && cd /content/btop && make && make install && \
    adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home
    
USER camenduru

RUN pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 torchtext==0.18.0 torchdata==0.8.0 --extra-index-url https://download.pytorch.org/whl/cu124 && \
    pip install xformers==0.0.28.post3 && \
    pip install opencv-python imageio imageio-ffmpeg ffmpeg-python av runpod && \
    pip install transformers diffusers accelerate peft opencv-python protobuf sentencepiece optimum-quanto && \
    git clone https://github.com/Yuanshi9815/OminiControl /content/OminiControl && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/black-forest-labs/FLUX.1-schnell/raw/main/scheduler/scheduler_config.json -d /content/model/scheduler -o scheduler_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/black-forest-labs/FLUX.1-schnell/raw/main/text_encoder/config.json -d /content/model/text_encoder -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/text_encoder/model.safetensors -d /content/model/text_encoder -o model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/black-forest-labs/FLUX.1-schnell/raw/main/text_encoder_2/config.json -d /content/model/text_encoder_2 -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/text_encoder_2/model-00001-of-00002.safetensors -d /content/model/text_encoder_2 -o model-00001-of-00002.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/text_encoder_2/model-00002-of-00002.safetensors -d /content/model/text_encoder_2 -o model-00002-of-00002.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/black-forest-labs/FLUX.1-schnell/raw/main/text_encoder_2/model.safetensors.index.json -d /content/model/text_encoder_2 -o model.safetensors.index.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/black-forest-labs/FLUX.1-schnell/raw/main/tokenizer/merges.txt -d /content/model/tokenizer -o merges.txt && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/black-forest-labs/FLUX.1-schnell/raw/main/tokenizer/special_tokens_map.json -d /content/model/tokenizer -o special_tokens_map.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/black-forest-labs/FLUX.1-schnell/raw/main/tokenizer/tokenizer_config.json -d /content/model/tokenizer -o tokenizer_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/black-forest-labs/FLUX.1-schnell/raw/main/tokenizer/vocab.json -d /content/model/tokenizer -o vocab.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/black-forest-labs/FLUX.1-schnell/raw/main/tokenizer_2/special_tokens_map.json -d /content/model/tokenizer_2 -o special_tokens_map.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/tokenizer_2/spiece.model -d /content/model/tokenizer_2 -o spiece.model && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/black-forest-labs/FLUX.1-schnell/raw/main/tokenizer_2/tokenizer.json -d /content/model/tokenizer_2 -o tokenizer.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/black-forest-labs/FLUX.1-schnell/raw/main/tokenizer_2/tokenizer_config.json -d /content/model/tokenizer_2 -o tokenizer_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/black-forest-labs/FLUX.1-schnell/raw/main/transformer/config.json -d /content/model/transformer -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/transformer/diffusion_pytorch_model-00001-of-00003.safetensors -d /content/model/transformer -o diffusion_pytorch_model-00001-of-00003.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/transformer/diffusion_pytorch_model-00002-of-00003.safetensors -d /content/model/transformer -o diffusion_pytorch_model-00002-of-00003.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/transformer/diffusion_pytorch_model-00003-of-00003.safetensors -d /content/model/transformer -o diffusion_pytorch_model-00003-of-00003.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/black-forest-labs/FLUX.1-schnell/raw/main/transformer/diffusion_pytorch_model.safetensors.index.json -d /content/model/transformer -o diffusion_pytorch_model.safetensors.index.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/black-forest-labs/FLUX.1-schnell/raw/main/vae/config.json -d /content/model/vae -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/vae/diffusion_pytorch_model.safetensors -d /content/model/vae -o diffusion_pytorch_model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/black-forest-labs/FLUX.1-schnell/raw/main/model_index.json -d /content/model -o model_index.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Yuanshi/OminiControl/resolve/main/omini/subject_512.safetensors -d /content/lora -o subject_512.safetensors

COPY ./worker_runpod.py /content/OminiControl/worker_runpod.py
WORKDIR /content/OminiControl
CMD python worker_runpod.py