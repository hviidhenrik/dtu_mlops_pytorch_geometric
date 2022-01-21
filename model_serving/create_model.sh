#!/bin/bash
torch-model-archiver --model-name energy_predictor --version 1.0 --serialized-file ./trained_jit_model.pt --export-path . --handler molecule_handler.py
