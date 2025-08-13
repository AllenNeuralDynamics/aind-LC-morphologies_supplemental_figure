# hash:sha256:a987f6cece6626ebc1da4e24e6dae08c53b96300f816ce08ab2af875cb099006
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN conda install -y \
        hdbscan==0.8.33 \
        jupyterlab==4.0.9 \
        k3d==2.16.0 \
    && conda clean -ya

RUN pip3 install -U --no-cache-dir \
    lc-reconstruction-analysis==0.0.24 \
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
RUN /postInstall
