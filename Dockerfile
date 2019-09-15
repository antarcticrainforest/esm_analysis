# ubuntu:latest seems to be LTS, i.e. 16.04 at the moment
FROM ubuntu:latest
RUN apt-get update -y && apt-get install -y python3 python3-pip git
RUN apt-get install cdo python3-cdo
RUN pip3 install --upgrade pip
RUN pip3 install dask distributed
# use /io to mount host file system later
RUN mkdir /io
WORKDIR /io
