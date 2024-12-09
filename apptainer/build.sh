#!/bin/bash
cd "$(dirname "$0")"
sudo apptainer build ../environment.sif environment.def