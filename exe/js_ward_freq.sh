#!/bin/bash

python ../scripts/ngram/js_ward_freq.py \
../2019-2025-data/past-shifts \
../2019-2025-data/group-settings \
--start-year 2020 \
--end-year 2025 \
--nmin 1 \
--nmax 5 \
--alpha 1 \
--laplace-support observed_ab \
--label-lang en
