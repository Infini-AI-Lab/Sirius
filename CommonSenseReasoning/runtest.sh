# accelerate launch --main_process_port 29502 --num_processes 4 main.py --tasks csqa --model meta-llama/Meta-Llama-3-8B-Instruct --griffin=True --check=True --kernel_size 16 --spr 0.5 --thr 0.1 
# accelerate launch --main_process_port 29501 --num_processes 8 main.py --tasks aqua --model meta-llama/Meta-Llama-3-8B-Instruct --shotfive --cats --check --kernel_size 16 
# accelerate launch --main_process_port 29501 --num_processes 8 main.py --tasks aqua --model meta-llama/Meta-Llama-3-8B-Instruct --shotfive --griffin 
# accelerate launch --main_process_port 29501 --num_processes 8 main.py --tasks aqua --model meta-llama/Meta-Llama-3-8B-Instruct --shotfive --griffin --check --kernel_size 16 
# python main.py --tasks aqua --model meta-llama/Meta-Llama-3-8B-Instruct --shotfive --griffin --check --kernel_size 16 --widthtree 8 
# accelerate launch --main_process_port 29501 --num_processes 6 main.py --tasks aqua --model meta-llama/Meta-Llama-3-8B-Instruct --shotfive --griffin --check --kernel_size 16 --widthtree 1 
# accelerate launch --main_process_port 29501 --num_processes 6 main.py --tasks aqua --model meta-llama/Meta-Llama-3-8B-Instruct --shotfive --griffin --check --kernel_size 16 --widthtree 2 
# accelerate launch --main_process_port 29501 --num_processes 6 main.py --tasks aqua --model meta-llama/Meta-Llama-3-8B-Instruct --shotfive --griffin --check --kernel_size 16 --widthtree 4 
# accelerate launch --main_process_port 29501 --num_processes 8 main.py --tasks strategyqa,sports,date --model meta-llama/Meta-Llama-3-8B-Instruct --shotfive 
# accelerate launch --main_process_port 29501 --num_processes 8 main.py --tasks strategyqa,sports,date --model meta-llama/Meta-Llama-3-8B-Instruct --shotfive --cats --spr 0.4 
# accelerate launch --main_process_port 29501 --num_processes 8 main.py --tasks strategyqa,sports,date --model meta-llama/Meta-Llama-3-8B-Instruct --shotfive --griffin --spr 0.4 
# accelerate launch --main_process_port 29501 --num_processes 8 main.py --tasks aqua --model meta-llama/Meta-Llama-3-8B-Instruct --shotfive 
# accelerate launch --main_process_port 29501 --num_processes 8 main.py --tasks aqua --model meta-llama/Meta-Llama-3-8B-Instruct --shotfive --cats 
# accelerate launch --main_process_port 29501 --num_processes 8 main.py --tasks aqua --model meta-llama/Meta-Llama-3-8B-Instruct --shotfive --griffin 
# accelerate launch --main_process_port 29510 --num_processes 8 main.py --cats --tasks aqua --model meta-llama/Meta-Llama-3-8B-Instruct --check --kernel_size 16 --spr 0.5 --thr 0.1 --shotfive --patternstrict --widthtree 4 
# accelerate launch --main_process_port 29510 --num_processes 8 main.py --griffin --tasks aqua --model meta-llama/Meta-Llama-3-8B-Instruct --check --kernel_size 16 --spr 0.5 --thr 0.1 --shotfive --widthtree 4 
# accelerate launch --main_process_port 29510 --num_processes 8 main.py --cats --tasks aqua --model meta-llama/Meta-Llama-3-8B-Instruct --check --kernel_size 16 --spr 0.5 --thr 0.1 --shotfive --patternstrict --widthtree 6 
# accelerate launch --main_process_port 29510 --num_processes 8 main.py --griffin --tasks aqua --model meta-llama/Meta-Llama-3-8B-Instruct --check --kernel_size 16 --spr 0.5 --thr 0.1 --shotfive --widthtree 6 
# accelerate launch --main_process_port 29510 --num_processes 8 main.py --cats --tasks aqua --model meta-llama/Meta-Llama-3-8B-Instruct --check --kernel_size 16 --spr 0.5 --thr 0.1 --shotfive --patternstrict --widthtree 8 
# accelerate launch --main_process_port 29510 --num_processes 8 main.py --griffin --tasks aqua --model meta-llama/Meta-Llama-3-8B-Instruct --check --kernel_size 16 --spr 0.5 --thr 0.1 --shotfive --widthtree 8 
# accelerate launch --main_process_port 29501 --num_processes 8 main.py --tasks aqua --model meta-llama/Meta-Llama-3-8B --shotfive --cats 
# accelerate launch --main_process_port 29501 --num_processes 8 main.py --tasks aqua --model meta-llama/Meta-Llama-3-8B --shotfive --griffin 
# accelerate launch --main_process_port 29501 --num_processes 8 main.py --tasks aqua --model meta-llama/Meta-Llama-3-8B-Instruct --shotfive --cats --check --kernel_size 16 --spr 0.5 --thr 0.1 --widthtree 1 
# accelerate launch --main_process_port 29501 --num_processes 8 main.py --tasks aqua --model meta-llama/Meta-Llama-3-8B --shotfive --cats --check --kernel_size 16 --spr 0.5 --thr 0.1 --widthtree 1 
# accelerate launch --main_process_port 29501 --num_processes 8 main.py --tasks aqua --model meta-llama/Meta-Llama-3-8B-Instruct --shotfive --griffin --check --kernel_size 14 --spr 0.5 --thr 0.1 --widthtree 1 
# accelerate launch --main_process_port 29501 --num_processes 8 main.py --tasks aqua --model meta-llama/Meta-Llama-3-8B-Instruct --shotfive --griffin --check --kernel_size 14 --spr 0.5 --thr 0.1 --widthtree 4 
# accelerate launch --main_process_port 29501 --num_processes 8 main.py --tasks aqua --model meta-llama/Meta-Llama-3-8B-Instruct --shotfive --griffin --check --kernel_size 12 --spr 0.5 --thr 0.1 --widthtree 1 

# treesizes=(1 4 6 8)
# for treesize in "${treesizes[@]}"
# do
# accelerate launch --main_process_port 29501 --num_processes 8 main.py --tasks csqa --model meta-llama/Meta-Llama-3-8B-Instruct --shotfive --cats --check --kernel_size 16 --spr 0.5 --thr 0.05 --widthtree $treesize
# done

# accelerate launch --main_process_port 29501 --num_processes 4 main.py --tasks date --model meta-llama/Llama-2-13b-chat-hf --shotfive --griffin --check --kernel_size 16 --spr 0.5 --thr 0.1 --widthtree 4 

# accelerate launch --main_process_port 29501 --num_processes 4 main.py --tasks sports --model meta-llama/Llama-2-13b-chat-hf --shotfive --griffin --check --kernel_size 16 --spr 0.5 --thr 0.1  --widthtree 4 
# accelerate launch --main_process_port 29501 --num_processes 4 main.py --tasks sports --model meta-llama/Llama-2-13b-chat-hf --shotfive --griffin --check --kernel_size 12 --spr 0.5 --thr 0.1  --widthtree 4 
treesize=(4 6 8) 
for treesizee in ${treesize[@]}; do 
accelerate launch --main_process_port 29501 --num_processes 6 main.py --tasks csqa --model meta-llama/Meta-Llama-3-8B-Instruct --shotfive --cats --check --kernel_size 16 --spr 0.5 --thr 0.05  --widthtree $treesizee --patternstrict 
done 
# accelerate launch --main_process_port 29501 --num_processes 6 main.py --tasks sports --model meta-llama/Meta-Llama-3-8B --shotfive --cats --check --kernel_size 16 --spr 0.5 --thr 0.1  --widthtree 1 --patternstrict 
# accelerate launch --main_process_port 29501 --num_processes 4 main.py --tasks sports --model meta-llama/Llama-2-13b-chat-hf --shotfive --griffin --check --kernel_size 12 --spr 0.5 --thr 0.05  --widthtree 4 
# accelerate launch --main_process_port 29501 --num_processes 8 main.py --tasks aqua --model meta-llama/Llama-2-7b-hf --shotfive --griffin --check --kernel_size 16 --spr 0.5 --thr 0.01 --widthtree 6 

# accelerate launch --main_process_port 29501 --num_processes 8 main.py --tasks aqua --model meta-llama/Meta-Llama-3-8B-Instruct --shotfive --griffin --check --kernel_size 12 --spr 0.5 --thr 0.1 --widthtree 8 
# accelerate launch --main_process_port 29501 --num_processes 8 main.py --tasks aqua --model meta-llama/Meta-Llama-3-8B-Instruct --shotfive --griffin --check --kernel_size 12 --spr 0.5 --thr 0.05 --widthtree 1 
# accelerate launch --main_process_port 29501 --num_processes 8 main.py --tasks aqua --model meta-llama/Meta-Llama-3-8B-Instruct --shotfive --griffin --check --kernel_size 12 --spr 0.5 --thr 0.01 --widthtree 1 
# accelerate launch --main_process_port 29501 --num_processes 8 main.py --tasks aqua --model meta-llama/Meta-Llama-3-8B --shotfive --griffin --check --kernel_size 16 --spr 0.5 --thr 0.1 --widthtree 1 
# accelerate launch --main_process_port 29501 --num_processes 8 main.py --tasks strategyqa --model meta-llama/Llama-2-7b-chat-hf --shotfive 
# accelerate launch --main_process_port 29501 --num_processes 8 main.py --tasks strategyqa --model meta-llama/Llama-2-7b-chat-hf --shotfive --cats --spr 0.5
# accelerate launch --main_process_port 29501 --num_processes 8 main.py --tasks strategyqa --model meta-llama/Llama-2-7b-chat-hf --shotfive --griffin --spr 0.5
# accelerate launch --main_process_port 29501 --num_processes 8 main.py --tasks aqua --model meta-llama/Llama-2-7b-chat-hf --shotfive --cats --check --kernel_size 16 --spr 0.5 --thr 0.01 --widthtree 6 --patternstrict 
# accelerate launch --main_process_port 29501 --num_processes 8 main.py --tasks aqua --model meta-llama/Llama-2-7b-chat-hf --shotfive --cats --check --kernel_size 16 --spr 0.5 --thr 0.05 --widthtree 1 
# accelerate launch --main_process_port 29501 --num_processes 8 main.py --tasks aqua --model meta-llama/Llama-2-7b-chat-hf --shotfive --griffin --check --kernel_size 16 --spr 0.5 --thr 0.01 --widthtree 6 
# accelerate launch --main_process_port 29501 --num_processes 8 main.py --tasks aqua --model meta-llama/Llama-2-7b-chat-hf --shotfive --griffin --check --kernel_size 16 --spr 0.5 --thr 0.05 --widthtree 1 
# accelerate launch --main_process_port 29501 --num_processes 8 main.py --tasks strategyqa --model meta-llama/Llama-2-7b-chat-hf --shotfive --cats --check --kernel_size 16 --spr 0.4 --thr 0.01 --widthtree 1 
# accelerate launch --main_process_port 29501 --num_processes 8 main.py --tasks strategyqa --model meta-llama/Llama-2-7b-chat-hf --shotfive --griffin --check --kernel_size 10 --spr 0.4 --thr 0.0005 --widthtree 1 
# accelerate launch --main_process_port 29501 --num_processes 8 main.py --tasks csqa --model meta-llama/Meta-Llama-3-8B-Instruct --shotfive --griffin --check --kernel_size 16 --spr 0.4 --thr 0.05 --widthtree 8 --patternstrict 
# accelerate launch --main_process_port 29501 --num_processes 8 main.py --tasks csqa,strategyqa,sports,date --model meta-llama/Meta-Llama-3-8B --shotfive --griffin

# treewidth=(1 4)
# for treew in ${treewidth[@]}; do
#     accelerate launch --main_process_port 29501 --num_processes 8 main.py --tasks csqa,strategyqa,sports,date --model meta-llama/Meta-Llama-3-8B-Instruct --shotfive --cats --check --kernel_size 16 --spr 0.4 --widthtree $treew --patternstrict --filteractiveenabled 
# done

# treewidth=(1 4)
# for treew in ${treewidth[@]}; do
#     accelerate launch --main_process_port 29501 --num_processes 8 main.py --tasks csqa,strategyqa,sports,date --model meta-llama/Meta-Llama-3-8B-Instruct --shotfive --griffin --check --kernel_size 16 --spr 0.4 --widthtree $treew --filteractiveenabled 
# done
# accelerate launch --main_process_port 29501 --num_processes 8 main.py --tasks aqua --model meta-llama/Meta-Llama-3-8B-Instruct --shotfive --griffin 
# accelerate launch --main_process_port 29501 --num_processes 6 main.py --tasks aqua --model meta-llama/Meta-Llama-3-8B-Instruct --shotfive --griffin --check --kernel_size 16 --widthtree 4 
# python main.py --tasks aqua --model meta-llama/Meta-Llama-3-8B-Instruct --shotfive --cats 
# python main.py --tasks aqua --model meta-llama/Meta-Llama-3-8B-Instruct --shotfive 
# python main.py --tasks aqua --model meta-llama/Llama-2-7b-chat-hf --shotfive 
