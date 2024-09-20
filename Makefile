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

.ONESHELL:
SHELL := /bin/bash
PYTHON3 := python3

setup: .venv

.venv:
	test -d .venv || $(PYTHON3) -m venv .venv --system-site-packages
	. .venv/bin/activate; \
	.venv/bin/pip install --upgrade pip; \
	.venv/bin/pip install --ignore-installed poetry; \
	export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring; \
	.venv/bin/poetry install

clean: clean-pyc
	rm -rf .venv
	rm -f poetry.lock
	rm -rf build
	rm -f gui-tool.spec

clean-pyc:
	find . -name "__pycache__" -exec rm -fr {} +
	find . -name ".pytest_cache" -exec rm -fr {} +
	find . -name ".coverage" -exec rm -fr {} +

test: .venv
	PYTHONPATH=. .venv/bin/poetry run pytest --junitxml=pytest.xml -m 'not slow' | tee pytest-coverage.txt
	make clean-pyc > /dev/null

lint: .venv
	. .venv/bin/activate; \
	isort .; \
	black .; \
	flake8

# BUILD SDK
UNIFY_SDK_VERSION ?= 0.3.5
build: .venv
	. .venv/bin/activate; \
	.venv/bin/poetry version $(UNIFY_SDK_VERSION); \
	.venv/bin/poetry build