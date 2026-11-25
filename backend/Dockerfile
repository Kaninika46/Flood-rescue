# Image with the "flood" conda env for the API and the RQ worker (GPU via the NVIDIA runtime).
FROM mambaorg/micromamba:2.0.5
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN CONDA_OVERRIDE_CUDA=12.9 micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes
WORKDIR /app
ENV PYTHONPATH=/app
