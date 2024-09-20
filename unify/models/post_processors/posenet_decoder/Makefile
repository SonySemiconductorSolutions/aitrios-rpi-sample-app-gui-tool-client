#
# Copyright 2024 Sony Semiconductor Solutions Corp. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 

SRC_CPP := posenet_binding.cpp posenet_decoder.cpp

CXX	:= g++

CXXFLAGS := \
	-pipe \
	-W \
	-Wall \
	-Wextra \
	-O3 \
	-std=c++17 \
	-fPIC \
	-shared \
	-I . \

ARCH := $(shell dpkg --print-architecture)
ifeq ($(filter $(ARCH),amd64 arm64),)
    $(error Unsupported architecture: $(ARCH))
endif

all: clean binding

binding:
	$(CXX) $(CXXFLAGS) -o libposenet_$(ARCH).so $^ $(SRC_CPP)

clean:
	rm -f libposenet_$(ARCH).so