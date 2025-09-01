# Let's start from an official nvidia cuda docker image to interact with NVIDIA
# container toolkit - feel free to update the tag as needed.
FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04

# Install packages here
RUN apt-get update && apt-get -y install --no-install-recommends adduser \
  ca-certificates wget unzip && \
  apt-get clean autoclean && \
  apt-get autoremove

# Install Miniconda (pinned version, edit as needed)
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py39_25.5.1-1-Linux-x86_64.sh -O ./miniconda-Linux-x86_64.sh && \
  bash ./miniconda-Linux-x86_64.sh -b -p $CONDA_DIR && \
  rm ./miniconda-Linux-x86_64.sh && \
  conda clean -afy

# Copy and create conda environment
COPY environment.yml .
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
  && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN conda env create -f environment.yml && conda clean -afy

# Create group and user to avoid permissions issues with local user/group
# when editing files in and out of docker container.
# Note: GNU/Linux systems assign the default 1000 User Identifier (UID) and
# Group Identifier (GID) to the first account created during installation. It is
# possible that your local UID and GID on your machine may be different, in that
# case you should edit the values in the commands below.
# You can see your UID and GID(s) by executing: `id`
# RUN addgroup --gid 1000 groupname
# RUN adduser --disabled-password --gecos "" --uid 1000 --gid 1000 username
# ENV HOME=/home/username
#USER username

# The base image already contains an user with UID=1000 named ubuntu
USER ubuntu


# Download datasets
WORKDIR /dx-privacy-curse-data
RUN wget -O /dx-privacy-curse-data/glove.6B.zip 'https://nlp.stanford.edu/data/glove.6B.zip'
RUN wget -O /dx-privacy-curse-data/glove.twitter.27B.zip 'https://nlp.stanford.edu/data/glove.twitter.27B.zip'
RUN wget -O /dx-privacy-curse-data/wiki.en.vec 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec'
RUN wget -O /dx-privacy-curse-data/GoogleNews-vectors-negative300.bin.gz 'https://drive.usercontent.google.com/download?export=download&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&confirm=t'

RUN unzip ./glove.6B.zip
RUN rm ./glove.6B.zip

RUN unzip ./glove.twitter.27B.zip
RUN rm ./glove.twitter.27B.zip

RUN gzip -d GoogleNews-vectors-negative300.bin.gz