FROM python:3.8-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt install -y wget && \
    apt install -y git && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN git clone -b cloud_training https://github.com/hviidhenrik/dtu_mlops_pytorch_geometric.git
WORKDIR /dtu_mlops_pytorch_geometric

RUN pip install torch==1.10.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html --no-cache-dir && \
    pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cpu.html --no-cache-dir && \
    pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cpu.html --no-cache-dir && \
    pip install torch-cluster -f https://data.pyg.org/whl/torch-1.10.0+cpu.html --no-cache-dir && \
    pip install torch-geometric --no-cache-dir && \
    pip install -e . --no-cache-dir && \
    pip install -r requirements.txt --no-cache-dir 
RUN wandb login daf73ad785871e297c736ddfb67826aa1663e305 #service account API key

WORKDIR /root
RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    rm -rf /root/tools/google-cloud-sdk/.install/.backup

ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

WORKDIR /dtu_mlops_pytorch_geometric

ENTRYPOINT ["python", "-u", "src/models/train.py", "--conf", "config/example.yaml"]
