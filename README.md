Running in a cloud server

sudo snap install astral-uv --classic

mkdir earth2studio-project && cd earth2studio-project
uv init --python=3.12
uv add "earth2studio @ git+https://github.com/NVIDIA/earth2studio.git@0.9.0"

```bash

python run_corrdiff.py
```

This will generate several .npy files given the initialization datetime and forecast hours.

# Deploy Corrdiff and run inference

## Deploy Corrdiff (docker)

```bash

docker login nvcr.io
username: $oauthtoken
password: $YOUR_API_KEY

docker pull nvcr.io/nim/nvidia/corrdiff:1.0.0
```

## Deploy Corrdiff (Singularity)

```bash
export SINGULARITY_DOCKER_USERNAME='$oauthtoken'
export SINGULARITY_DOCKER_PASSWORD='YOUR_API_KEY'

singularity pull docker://nvcr.io/nim/nvidia/corrdiff:1.0.0
```
