#!/bin/bash

MODELS=(
    "gpt-4o"
    "qwen-vl-max"
    "llava-next-7b"
    "llava-next-34b"
    # "gemini"
    "minicpm"
    "llama"
)
DATASETS=(
    "Hard"
    "Normal"
)
OUTPUT_DIR="results/zero-shot"
TASKS=(
    "step_type"
    "description_correct"
    "description_error_type"
    "description_relevant"
    "logic_relevant"
    "logic_correct"
    "logic_error_type"
    "informativeness"
    "cot"
)

run_evaluation() {
    local MODEL=$1
    local DATASET=$2
    local TASK=$3
    local ITERATION=$4
    local SAMPLES=$5
    local OUTPUT_PATH="${OUTPUT_DIR}/${MODEL}/${DATASET}/${MODEL}_${TASK}_iteration${ITERATION}.jsonl"
    python eval.py --eval_model "$MODEL" --eval_dataset "$DATASET" --eval_task "$TASK" --output_path "$OUTPUT_PATH" --setting zero-shot
}

for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for TASK in "${TASKS[@]}"; do
            for i in {1..3}; do
                run_evaluation "$MODEL" "$DATASET" "$TASK" "$i"
            done
        done
    done
done
