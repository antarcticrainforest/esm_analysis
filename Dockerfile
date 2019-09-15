FROM ubuntu:latest

# Install script dependance avaible on apt source
RUN /bin/bash -c "apt-get update && apt-get install -y --allow-unauthenticated \
                wget vim \
                zip bzip2 \
                gcc g++ gfortran \
                build-essential \
                cdo \
                python3-cdo \
                python3-netcdf4 \
                libnetcdf-dev"

# use /io to mount host file system later
# RUN mkdir /root
WORKDIR /root
