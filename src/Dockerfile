FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONIOENCODING=utf-8 \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  DEBIAN_FRONTEND=noninteractive

# Install necessary packages and clean up
RUN apt-get update && apt-get install -y --no-install-recommends \
  git \
  curl \
  wget \
  gnupg2 \
  apt-transport-https \
  ca-certificates \
  software-properties-common \
  build-essential \
  procps \
  lsb-release \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Create a non-root user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
  && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
  && apt-get update \
  && apt-get install -y sudo \
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
  && chmod 0440 /etc/sudoers.d/$USERNAME \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Set up Python development tools
RUN pip install --no-cache-dir --upgrade pip

# Set the working directory
WORKDIR /workspace

# Install Python packages from requirements.txt when present
COPY requirements.txt* ./
RUN if [ -f "requirements.txt" ]; then pip install --no-cache-dir -r requirements.txt; fi

# Switch to non-root user
USER $USERNAME