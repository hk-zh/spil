#!/bin/bash

cd calvin/tacto
pip install -e .
cd ..
pip install -e .
cd skill_generator
pip install -e .
cd ..
pip install -e .

