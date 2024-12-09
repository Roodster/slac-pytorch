#!/bin/bash
apptainer run --nv --bind $(pwd):/app environment.sif

