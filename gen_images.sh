#!/bin/bash

for i in {0..500}; do & python3 generate_data.py --hd5 MuonData/hd5/output_${i}.hd5 --json MuonData/json/pipe_${i}.json; done

