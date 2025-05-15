input_path='/data/framework_vllm/cxl/CMB/CMB/data/CMB/CMB-Exam/CMB-test/CMB-test-choice-question-merge.json'   # CMB-Exam
# input_path='./data/CMB-Clin/CMB-Clin-qa.json'                             # CMB-Clin


task_name='Exam' 

model_id="cxl" # which model to evaluate
mkdir -p logs/${task_name}/
CUDA_VISIBLE_DEVICES=0

accelerate launch --main_process_port 27274 --config_file ./configs/accelerate_config.yaml  ./src/generate_answers.py \
--model_id=$model_id \
--all_gather_freq=1 \
--input_path=$input_path \
--output_path=./result/${task_name}/${model_id}/modelans4.json \
--batch_size 50 \
--model_config_path="./configs/model_config.yaml" \
--use_eagle \
--use_cot 
#跑eagle模式batch_size不能特别大，会报错