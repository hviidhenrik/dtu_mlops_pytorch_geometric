#!/bin/bash
#cat config.properties
#service_envelope=json

torchserve --start --ncs --model-store . --models energy_predictor=energy_predictor.mar # --ts-config ./config.properties
