#!/bin/bash

python ../scripts/ngram/js_past_vs_found_freq.py \
../2019-2025-data/past-shifts/GCU.lp \
../2019-2025-data/group-settings/GCU/ \
../found-model/GCU/2024-10-13/ \
--pstart-year 2020 \
--pend-year 2025 \
--nmin 1 \
--nmax 5 \
--laplacek 1 \
--past-mode summary \
--fdate-from 20241013 \
--fdate-to 20241113 \
--past-base loo
