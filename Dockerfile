# ubuntu:latest seems to be LTS, i.e. 16.04 at the moment
FROM ubuntu:18.04
RUN apt-get update -y && \ 
 apt-get install -y --no-install-recommends python3=3.6 python3-pip git && \
 apt-get clean && \
 rm -rf /var/lib/apt/lists/*
RUN pip3 install --upgrade pip
# use /io to mount host file system later
RUN mkdir /io
WORKDIR /io
