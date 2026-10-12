#!/bin/bash

DATASET="ShareGPT_V3_unfiltered_cleaned_split.json"
TOKENIZER="Llama-3.2-1B-Instruct/"

REQUEST_RATES=(3 4 5 6 7 )

for rate in ${REQUEST_RATES[@]}
do
    echo "=============================="
    echo "Running benchmark with request-rate=${rate}"
    echo "=============================="

    python test/benchmark/service/benchmark_sharegpt.py \
        --dataset ${DATASET} \
        --tokenizer ${TOKENIZER} \
        --num-prompts 4000 \
        --request-rate ${rate}

    echo ""
done
