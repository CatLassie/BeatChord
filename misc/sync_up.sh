#!/usr/bin/env bash

rsync -azP --exclude '*/venv' --exclude '*/.idea' --exclude '*/.git' --exclude '*/__pycache__' --exclude '*/archive' --exclude '*/output' --include '*/' --include '*.py' --include '*.sh' --exclude '*' ../drum_pt $1:~/python-workspace
