# hash:sha256:36928a3bb2e99b5e0d445378259f5d82052225a2e1d5998d1c0e3318f21bec91
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN pip3 install -U --no-cache-dir \
    brainglobe-atlasapi==2.1.0 \
    k3d==2.16.0 \
    lc-reconstruction-analysis==0.1.0 \
    matplotlib==3.8.2 \
    networkx==3.2.1 \
    numpy==1.26.2 \
    pandas==2.1.3 \
    scikit-learn==1.3.2 \
    scipy==1.11.4 \
    seaborn==0.13.0 \
    tqdm==4.66.1 \
    umap-learn==0.5.5

COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall
