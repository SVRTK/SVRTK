#
# SVRTK : SVR reconstruction based on MIRTK
#
# Copyright 2018-2022 King's College London
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# Set up a docker image for: https://github.com/SVRTK/SVRTK
#
# Build with
#
#   docker build -t <name> .
#
# In the case of a proxy (located at 192.168.13.14:3128), do:
#
#    docker build --build-arg http_proxy=http://10.41.13.4:3128 --build-arg https_proxy=https://10.41.13.6:3128 -t svrtk .
#
# --no-cache will force a clean build
# 
# To run an interactive shell inside this container, do:
#
#   docker run -it fetalrecon /bin/bash 
#
#   docker run -it --mount type=bind,source=file_path_on_your_computer,target=/data svrtk
# 


FROM ubuntu:20.04

# update and install dependencies
RUN         apt-get update \
                && apt-get install -y --no-install-recommends \
                    git cmake cmake-curses-gui python \
                    build-essential libtbb-dev libboost-all-dev libeigen3-dev zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget zram-config ca-certificates

# Download MIRTK (please use this version with additional SVRTK compilation options)
RUN git clone https://github.com/SVRTK/MIRTK.git /home/

# Download SVRTK into /Packages folder of MIRTK
RUN git clone https://github.com/SVRTK/SVRTK.git /home/MIRTK/Packages/SVRTK

# Compile the code
RUN cd /usr/src/MIRTK \
&& mkdir build \
&& cd build/ \
&& cmake -D WITH_TBB="ON" -D MODULE_SVRTK="ON" .. \
&& make -j

# Copy 
RUN cp /home/MIRTK/build/bin/mirtk /usr/local/bin/

# update path
ENV PATH="/home/MIRTK/build/bin:/home/MIRTK/build/lib/tools:${PATH}"
