# accelerate launch  --num_processes 2 main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B,thresholdbased=True,check=False,spr=0.5 --tasks gsm8k --batch_size 1 --limit=20 
# accelerate launch  --num_processes 2 main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,thresholdbased=True,check=False,spr=0.5 --tasks gsm8k --batch_size 1 --limit 0.3 
# accelerate launch --num_processes 8 main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,cats=True,widthtree=2,check=False --tasks gsm8k --batch_size 1 --limit 0.2 
# accelerate launch --num_processes 8 main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,cats=True,check=False --tasks gsm8k --batch_size 1 --limit 0.2 
# accelerate launch --num_processes 8 main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,cats=True,widthtree=1,check=True,kernel_size=16,spr=0.5,thr=0.1,patternstrict=True --tasks gsm8k --batch_size 1 --limit 0.2 
# accelerate launch --num_processes 4 main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,cats=True,check=True,kernel_size=16,spr=0.5,thr=0.1 --tasks gsm8k --batch_size 1 --limit 0.2 
# accelerate launch --num_processes 6 main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,cats=True,widthtree=2,check=True,kernel_size=16,spr=0.5,thr=0.1,patternstrict=True --tasks gsm8k --batch_size 1 --limit 10 

# accelerate launch --num_processes 6 main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,cats=True,check=True,kernel_size=16,spr=0.5,thr=0.1 --tasks gsm8k --batch_size 1 --limit 0.3 
# accelerate launch --main_process_port 29510 --num_processes 8 --num_machines 1 main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,cats=True,check=True,kernel_size=16,spr=0.5,thr=0.9 --tasks gsm8k --batch_size 1 
# accelerate launch --main_process_port 29510 --num_processes 8 --num_machines 1 main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,widthtree=1,griffin=True,check=True,kernel_size=16,spr=0.5,thr=0.1 --tasks gsm8k --batch_size 1 --limit 0.3 
# accelerate launch --main_process_port 29510 --num_processes 8 --num_machines 1 main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B,widthtree=2,cats=True,check=True,kernel_size=16,spr=0.5,thr=0.1,patternstrict=True,filteractiveenabled=True --tasks gsm8k --batch_size 1 --limit 0.3 

# sprss=(0.15 0.2 0.25) 
# for spars in "${sprss[@]}" 
# do 
#     accelerate launch --main_process_port 29510 --num_processes 6 --num_machines 1 main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,widthtree=6,cats=True,check=True,kernel_size=9,spr=$spars,thr=0.001,patternstrict=True --tasks gsm8k --batch_size 1 --limit 0.3 
#     accelerate launch --main_process_port 29510 --num_processes 6 --num_machines 1 main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,widthtree=6,cats=True,check=True,kernel_size=9,spr=$spars,thr=0.005,patternstrict=True --tasks gsm8k --batch_size 1 --limit 0.3 
#     accelerate launch --main_process_port 29510 --num_processes 6 --num_machines 1 main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,widthtree=6,cats=True,check=True,kernel_size=9,spr=$spars,thr=0.01,patternstrict=True --tasks gsm8k --batch_size 1 --limit 0.3 
# done 
# accelerate launch --main_process_port 29510 --num_processes 8 --num_machines 1 main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,widthtree=6,cats=True,check=True,kernel_size=12,spr=0.3,thr=0.0001,patternstrict=True --tasks gsm8k --batch_size 1 --limit 0.3 
# nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s none -o sparsemodelcall -f true -x true python main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,griffin=True,check=False --tasks gsm8k --batch_size 1 --limit 0.3 
# TORCH_LOGS="+dynamo" TORCHDYNAMO_VERBOSE=1 python main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,griffin=False,check=False,contextlength=1500,kernel_size=16,thr=0.05,attentionimplementation=general,widthtree=4 --tasks gsm8k_cot --batch_size 1 --limit 0.3 
# lengs=(1 2 4 8 16 32 64 96) 
# for leng in "${lengs[@]}" 
# do 
#     TOKENIZERS_PARALLELISM=false python getcompilego.py --length $leng 
# done 

# python main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,griffin=True,check=False,contextlength=1500,kernel_size=16,thr=0.05 --tasks gsm8k_cot --batch_size 1 --limit 0.3 
# python main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,griffin=True,check=False,contextlength=1500,kernel_size=16,thr=0.05,widthtree=4 --tasks gsm8k_cot --batch_size 1 --limit 0.3 

# accelerate launch --num_processes 10 -m lm_eval --model hf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct --tasks mmlu_flan_cot_fewshot --batch_size 1 --limit 0.05 
# treesizes=(1 4 6 8)
# for treesize in "${treesizes[@]}" 
# do 
# accelerate launch --main_process_port 29510 --num_processes 8 --num_machines 1 main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B,cats=True,check=True,kernel_size=16,widthtree=$treesize,patternstrict=True --tasks gsm8k --batch_size 1 
# done 
NCCL_DEBUG=INFO accelerate launch --main_process_port 29510 --num_processes 4 --num_machines 1 main.py --model xhf --model_args pretrained=meta-llama/Llama-2-13b-hf,griffin=True,check=True,kernel_size=16,widthtree=4,patternstrict=True,thr=0.1 --tasks gsm8k --batch_size 1 
# python main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B,cats=True,check=True,kernel_size=16,widthtree=1,patternstrict=True --tasks gsm8k --batch_size 1 
# accelerate launch --main_process_port 29512 --num_processes 8 --num_machines 1 main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,cats=True,check=True,kernel_size=16 --tasks gsm8k --batch_size 1 --limit 0.2 
# accelerate launch --main_process_port 29510 --num_processes 2 --num_machines 1 main.py --model xhf --model_args pretrained=meta-llama/Llama-2-7b-chat-hf,cats=True,check=False --tasks truthfulqa_gen --batch_size 1 
# python main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,griffin=True,check=False,contextlength=1024 --tasks gsm8k --batch_size 1 --limit 0.3 

# python main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,griffin=False,check=False,contextlength=2048 --tasks gsm8k --batch_size 1 --limit 0.3 
# python main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,griffin=True,check=False,contextlength=2048 --tasks gsm8k --batch_size 1 --limit 0.3 
# python main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,griffin=True,check=False --tasks gsm8k --batch_size 1 --limit 0.3 
# python main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,griffin=True,check=False --tasks gsm8k --batch_size 1 --limit 0.3 
# accelerate launch --main_process_port 29510 --num_processes 8 --num_machines 1 main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,widthtree=6,cats=True,check=True,kernel_size=12,spr=0.3,thr=0.0005,patternstrict=True --tasks gsm8k --batch_size 1 --limit 0.3 

# sprss12=(0.3 0.35)
# for spars in "${sprss12[@]}" 
# do 
#     accelerate launch --main_process_port 29510 --num_processes 6 --num_machines 1 main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,widthtree=6,cats=True,check=True,kernel_size=12,spr=$spars,thr=0.001,patternstrict=True --tasks gsm8k --batch_size 1 --limit 0.3 
#     accelerate launch --main_process_port 29510 --num_processes 6 --num_machines 1 main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,widthtree=6,cats=True,check=True,kernel_size=12,spr=$spars,thr=0.005,patternstrict=True --tasks gsm8k --batch_size 1 --limit 0.3 
#     accelerate launch --main_process_port 29510 --num_processes 6 --num_machines 1 main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,widthtree=6,cats=True,check=True,kernel_size=12,spr=$spars,thr=0.01,patternstrict=True --tasks gsm8k --batch_size 1 --limit 0.3 
# done 
# sprss=(0.2 0.3 0.4 0.5 0.6 0.7 0.8) 
# for spars in "${sprss[@]}" 
# do 
#     accelerate launch --main_process_port 29510 --num_processes 8 --num_machines 1 main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,widthtree=4,griffin=True,check=True,kernel_size=16,spr=$spars,thr=0.05 --tasks gsm8k --batch_size 1 --limit 0.3 
# done 
# accelerate launch --main_process_port 29510 --num_processes 8 --num_machines 1 main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,widthtree=4,griffin=True,check=True,kernel_size=9,spr=0.15,thr=0.05 --tasks gsm8k --batch_size 1 --limit 0.3 
# accelerate launch --main_process_port 29510 --num_processes 8 --num_machines 1 main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,widthtree=8,griffin=True,check=True,kernel_size=9,spr=0.15,thr=0.0005 --tasks gsm8k --batch_size 1 --limit 0.3 
# accelerate launch --main_process_port 29510 --num_processes 8 --num_machines 1 main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,widthtree=8,griffin=True,check=True,kernel_size=9,spr=0.2,thr=0.001 --tasks gsm8k --batch_size 1 --limit 0.3 
# accelerate launch --main_process_port 29510 --num_processes 8 --num_machines 1 main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,widthtree=8,griffin=True,check=True,kernel_size=9,spr=0.25,thr=0.01 --tasks gsm8k --batch_size 1 --limit 0.3 
# accelerate launch --main_process_port 29510 --num_processes 8 --num_machines 1 main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,widthtree=6,griffin=True,check=True,kernel_size=9,spr=0.3,thr=0.001 --tasks gsm8k --batch_size 1 --limit 0.3 
# accelerate launch --main_process_port 29510 --num_processes 8 --num_machines 1 main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,widthtree=6,griffin=True,check=True,kernel_size=12,spr=0.3,thr=0.005 --tasks gsm8k --batch_size 1 --limit 0.3 
# accelerate launch --main_process_port 29510 --num_processes 8 --num_machines 1 main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,widthtree=6,griffin=True,check=True,kernel_size=12,spr=0.3,thr=0.005 --tasks gsm8k --batch_size 1 --limit 0.3 
# accelerate launch --main_process_port 29510 --num_processes 8 --num_machines 1 main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,widthtree=4,griffin=True,check=True,kernel_size=16,spr=0.6,thr=0.05 --tasks gsm8k --batch_size 1 --limit 0.3 
# accelerate launch --main_process_port 29510 --num_processes 8 --num_machines 1 main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,widthtree=4,griffin=True,check=True,kernel_size=16,spr=0.6,thr=0.1 --tasks gsm8k --batch_size 1 --limit 0.3 
# accelerate launch --main_process_port 29510 --num_processes 8 --num_machines 1 main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,cats=True,check=True,kernel_size=16,spr=0.5,thr=0.1 --tasks gsm8k --batch_size 1 --limit 0.3 
# accelerate launch --main_process_port 29510 --num_processes 8 --num_machines 1 main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,widthtree=8,griffin=True,check=True,kernel_size=16,spr=0.5,thr=0.1 --tasks gsm8k --batch_size 1 --limit 0.3 
# accelerate launch --main_process_port 29510 --num_processes 8 --num_machines 1 main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,widthtree=4,cats=True,check=True,kernel_size=16,spr=0.5,thr=0.1,patternstrict=True --tasks gsm8k --batch_size 1 --limit 0.3 
# accelerate launch --main_process_port 29510 --num_processes 8 --num_machines 1 main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,widthtree=8,cats=True,check=True,kernel_size=16,spr=0.5,thr=0.1,patternstrict=True,filteractiveenabled=True --tasks gsm8k --batch_size 1 --limit 0.3 

# accelerate launch --main_process_port 29510 --num_processes 8 --num_machines 1 main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,widthtree=2,cats=True,check=True,kernel_size=16,spr=0.5,thr=0.1,patternstrict=True,filteractiveenabled=False --tasks gsm8k --batch_size 1 --limit 0.3 
# accelerate launch --main_process_port 29510 --num_processes 8 --num_machines 1 main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,widthtree=4,cats=True,check=True,kernel_size=16,spr=0.5,thr=0.1,patternstrict=True,filteractiveenabled=False --tasks gsm8k --batch_size 1 --limit 0.3 
# accelerate launch --main_process_port 29510 --num_processes 8 --num_machines 1 main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,widthtree=8,cats=True,check=True,kernel_size=16,spr=0.5,thr=0.1,patternstrict=True,filteractiveenabled=False --tasks gsm8k --batch_size 1 --limit 0.3 

# accelerate launch --num_processes 6 main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,cats=True,widthtree=2,check=True,kernel_size=16,spr=0.5,thr=0.1,patternstrict=True --tasks gsm8k --batch_size 1 --limit 0.3 
# accelerate launch --num_processes 6 main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,cats=True,widthtree=4,check=True,kernel_size=16,spr=0.5,thr=0.1,patternstrict=True --tasks gsm8k --batch_size 1 --limit 0.3 
# accelerate launch --num_processes 6 main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,cats=True,widthtree=8,check=True,kernel_size=16,spr=0.5,thr=0.1,patternstrict=True --tasks gsm8k --batch_size 1 --limit 0.3 
# accelerate launch --num_processes 6 main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,cats=True,widthtree=8,check=True,kernel_size=16,spr=0.5,thr=0.1,patternstrict=True --tasks gsm8k --batch_size 1 --limit 0.3 
# accelerate launch --num_processes 8 main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,cats=True,widthtree=4,check=True,kernel_size=16,spr=0.5,thr=0.1,patternstrict=True --tasks gsm8k --batch_size 1 --limit 0.2 
# accelerate launch --num_processes 8 main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,cats=True,widthtree=8,check=True,kernel_size=16,spr=0.5,thr=0.1,patternstrict=True --tasks gsm8k --batch_size 1 --limit 0.2 
# accelerate launch --num_processes 8 main.py --model xhf --model_args pretrained=meta-llama/Llama-2-7b-chat-hf,cats=True,widthtree=1,check=True,kernel_size=16,spr=0.5,thr=0.1 --tasks gsm8k --batch_size 1 --limit 0.2 
# accelerate launch --num_processes 8 main.py --model xhf --model_args pretrained=meta-llama/Llama-2-7b-chat-hf,cats=True,widthtree=2,check=True,kernel_size=16,spr=0.5,thr=0.1 --tasks gsm8k --batch_size 1 --limit 0.2 
# accelerate launch --num_processes 8 main.py --model xhf --model_args pretrained=meta-llama/Llama-2-7b-chat-hf,cats=True,widthtree=4,check=True,kernel_size=16,spr=0.5,thr=0.1 --tasks gsm8k --batch_size 1 --limit 0.2 
# accelerate launch --num_processes 8 main.py --model xhf --model_args pretrained=meta-llama/Llama-2-7b-chat-hf,cats=True,widthtree=8,check=True,kernel_size=16,spr=0.5,thr=0.1 --tasks gsm8k --batch_size 1 --limit 0.2 


# accelerate launch --num_processes 4 main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,cats=True,widthtree=4,check=True,kernel_size=16,spr=0.5,thr=0.1 --tasks gsm8k --batch_size 1 --limit 0.2 
# accelerate launch --num_processes 4 main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,cats=True,widthtree=8,check=True,kernel_size=16,spr=0.5,thr=0.1 --tasks gsm8k --batch_size 1 --limit 0.2 
# python main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,cats=True,widthtree=4,check=True,kernel_size=16,spr=0.5,thr=0.1 --tasks gsm8k --batch_size 1 --limit 0.2 
# accelerate launch --num_processes 8 main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,cats=True,check=True,kernel_size=16,spr=0.5,thr=0.1 --tasks gsm8k --batch_size 1 --limit 0.2 
# accelerate launch  --num_processes 2 main.py --model xhf --model_args pretrained=meta-llama/Meta-Llama-3-8B,thresholdbased=True,check=False,spr=0.3 --tasks gsm8k --batch_size 1 --limit=20 
