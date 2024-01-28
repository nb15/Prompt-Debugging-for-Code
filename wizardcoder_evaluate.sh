#!/bin/bash
#DATASET="humaneval mbpp"
DATASET="humaneval"
MODELS="wizard_7B"
#MODELS="wizard_7B wizard_15B"
NUM_DELTAS=7
RESULT=''

for DATA in $DATASET
do
    for MODEL in $MODELS
    do
        echo "Processing: " $DATA " " $MODEL "\n"
        memorization=$(python3 reports/process_wizardcoder.py --dataset $DATA --model $MODEL --run-id 1 --num-deltas $NUM_DELTAS)
        
        echo "Evaluation: " $DATA " " $MODEL "\n"
        for ((i = 0; i < $NUM_DELTAS; i++)); do
            echo "Starting Delta: " $i
            EVAL=$(python3 evaluate_deltas.py --dataset $DATASET --samples generated_code/deltas/${DATA}_${MODEL}_run1_Delta_${i}.jsonl)
            RESULT="$RESULT\n\n$DATA $MODEL Delta $i\nMemorization:$memorization\n$EVAL"
        done
        # python3 reports/process_wizardcoder.py --dataset humaneval --model wizard_7B --run-id 2

        # for ((i = 0; i < $NUM_DELTAS; i++)); do
        #     evalplus.evaluate --dataset $DATASET --samples /generated_code/deltas${DATA}_${MODEL}_run2_Delta_${i}.jsonl
        # done
    done

done
echo $RESULT
echo "$RESULT" >> wizardcoder_evaluation.txt