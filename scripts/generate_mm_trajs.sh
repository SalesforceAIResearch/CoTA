#!/bin/bash
#. /export/agentstudio-family/zixian/.bashrc

# export HF_HOME=/export/agentstudio-family/transformers/
# export ROOT_DIR="/export/agentstudio-family/zixian/mma"

## change for each dataset
gen=$2
dataset=$1
echo $1
echo $2

exp_id="0915-15-tools"
model="gpt-4o-2024-08-06"

## activate llava environment
source /export/share/zixianma/miniconda/bin/activate
conda activate /export/agentstudio-family/miniconda3/envs/mmall
source /export/agentstudio-family/zixian/.bashrc
if [[ $gen == 1 ]];
then
    cd /export/agentstudio-family/zixian/mma
    python -m mma.run_multimodal_agent --execute --max-reply 10 --exp-id $exp_id --model $model  --dataset $dataset --prompt-format json
fi
cd /export/agentstudio-family/zixian/VLMEvalKit-zixianma
python run_eval_on_preds.py --model gpt4o --data MMVet --result-file "/export/agentstudio-family/zixian/mma/prediction/${model}/${dataset}/${exp_id}-json-max-reply-10-seed-42.jsonl" 

cd /export/agentstudio-family/zixian/mma
python -m mma.preprocess_train_data --model $model --exp-id "${exp_id}-json-max-reply-10-seed-42" --dataset $dataset --save-llava-format 