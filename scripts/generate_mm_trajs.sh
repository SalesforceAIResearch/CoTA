#!/bin/bash
# change for each dataset
dataset=$1

exp_id="1221-test"
model="gpt-4o-2024-08-06"
code_dir="/export/agentstudio-family/zixian/taco/"

source /export/share/zixianma/miniconda/bin/activate /export/agentstudio-family/miniconda3/envs/mmall
source /export/agentstudio-family/zixian/.bashrc

# Generate
python ${code_dir}/taco/run_multimodal_agent.py --execute --max-reply 10 --exp-id $exp_id --model $model  --dataset $dataset --prompt-format cota

# Verify final answers
## The data argument determines whether to use exact matching or llm judge. Using MMVet to activate llm judge.
python ${code_dir}/VLMEvalKit/run_eval_on_preds.py --data MMVet --result-file "/export/agentstudio-family/zixian/cota/prediction/${model}/${dataset}/${exp_id}-cota-max-reply-10-seed-42.jsonl" 

# Parse and convert 
## convert positive examples into Mantis and/or LLaVA training data format and negative examples into Direct answer format with groundtruth answers. 
python cota/preprocess_train_data.py --model $model --exp-id "${exp_id}-cota-max-reply-10-seed-42" --dataset $dataset --save-llava-format 