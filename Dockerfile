# ubuntu:latest seems to be LTS, i.e. 16.04 at the moment
FROM ubuntu:18.04
RUN apt-get install -y --no-install-recommends python3=3.6 python3-pip=9.0.1-2.3~ubuntu1.18.04.1 git=1:2.17.1-1ubuntu0.5 && \
 apt-get clean && \
 rm -rf /var/lib/apt/lists/*
RUN pip3 install --upgrade pip==20.0.2
# use /io to mount host file system later
RUN mkdir /io
WORKDIR /io
