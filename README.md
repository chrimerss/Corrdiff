Running in a cloud server

sudo snap install astral-uv --classic

mkdir earth2studio-project && cd earth2studio-project
uv init --python=3.12
uv add "earth2studio @ git+https://github.com/NVIDIA/earth2studio.git@0.9.0"
