FROM mambaorg/micromamba:latest

WORKDIR /work


RUN micromamba install -y -n base -c conda-forge \
      python=3.12 pip clingo \
    && micromamba clean -a -y

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt

COPY . .

ENTRYPOINT ["/bin/bash"]
