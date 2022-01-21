FROM pytorch/torchserve:0.3.0-cpu

# COPY src /home/model-server/src
COPY model_serving/trained_jit_model.pt model_serving/molecule_handler.py /home/model-server/

USER root
RUN printf "\nservice_envelope=json" >> /home/model-server/config.properties
USER model-server

RUN pip install torch==1.10.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html --no-cache-dir && \
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cpu.html --no-cache-dir && \
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cpu.html --no-cache-dir && \
pip install torch-cluster -f https://data.pyg.org/whl/torch-1.10.0+cpu.html --no-cache-dir && \
pip install torch-geometric --no-cache-dir && \
pip install -r requirements.txt --no-cache-dir && \
pip install -e . --no-cache-dir


RUN torch-model-archiver \
  --model-name=equivariant_transformer \
  --version=1.0 \ # --model-file=/home/model-server/model.py \
  --serialized-file=/home/model-server/trained_jit_model.pt \
  --handler=/home/model-server/molecule_handler.py \
  --export-path=/home/model-server/model-store

CMD ["torchserve", \
     "--start", \
     "--ts-config=/home/model-server/config.properties", \
     "--models", \
     "equivariant_transformer=equivariant_transformer.mar"]
END

