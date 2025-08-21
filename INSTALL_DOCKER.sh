#!/bin/bash

# --- UPDATE AND INSTALL PREREQUISITES ---
echo "Updating package list and installing prerequisites..."
sudo apt-get update
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg

# --- ADD DOCKER'S OFFICIAL GPG KEY ---
echo "Adding Docker's GPG key..."
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# --- SET UP THE DOCKER REPOSITORY ---
echo "Setting up Docker's APT repository..."
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# --- INSTALL DOCKER ENGINE ---
echo "Updating package list and installing Docker Engine..."
sudo apt-get update
sudo apt-get install -y \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    docker-buildx-plugin \
    docker-compose-plugin

# --- VERIFY INSTALLATION ---
echo "Verifying Docker installation by running the hello-world container..."
sudo docker run hello-world

echo "Docker installation is complete."

# --- OPTIONAL: RUN DOCKER WITHOUT SUDO ---
# If you want to run Docker commands without prepending 'sudo',
# you can add your user to the 'docker' group.
# Uncomment the line below to do so.
#
# sudo usermod -aG docker $USER
#
# After running this command, you must log out and log back in
# for the group membership to be re-evaluated.
