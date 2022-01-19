FROM python:3.8-slim

RUN apt update && \
apt install --no-install-recommends -y build-essential gcc && \
apt install -y git && \
apt clean && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/hviidhenrik/dtu_mlops_pytorch_geometric.git
WORKDIR "/dtu_mlops_pytorch_geometric"

# Installs google cloud sdk, this is mostly for using gsutil to export model.
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
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup
# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

RUN pip install torch==1.10.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html --no-cache-dir && \
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cpu.html --no-cache-dir && \
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cpu.html --no-cache-dir && \
pip install torch-cluster -f https://data.pyg.org/whl/torch-1.10.0+cpu.html --no-cache-dir && \
pip install torch-geometric --no-cache-dir && \
pip install -e . --no-cache-dir && \
pip install -r requirements.txt --no-cache-dir

RUN wandb login daf73ad785871e297c736ddfb67826aa1663e305 #service account API key

# Bundle app source
COPY . /root

ENTRYPOINT ["python", "-u", "src/models/train.py", "--conf", "config/example.yaml", "--dataset", "QM9", "--log-dir", "output/"]
