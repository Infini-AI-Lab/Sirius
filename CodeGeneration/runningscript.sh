# accelerate launch --num_processes 8 main.py \
#   --model meta-llama/Llama-2-7b-chat-hf \
#   --tasks mbppplus \
#   --do_sample False \
#   --n_samples 1 \
#   --batch_size 1 \
#   --max_length_generation 512 \
#   --enable_epatches \
#   --griffin \
#   --allow_code_execution \
#   --spr 0.5 \
#   --widthtree 10 \
#   --check \
#   --kernelsize 12 \
#   --thr 0.05 \

# treesizes=(1 4 6 8) 
# for treesize in ${treesizes[@]} 
# do 
# accelerate launch --num_processes 8 main.py \
#   --model meta-llama/Llama-2-7b-chat-hf \
#   --tasks humaneval \
#   --do_sample False \
#   --n_samples 1 \
#   --batch_size 1 \
#   --max_length_generation 512 \
#   --enable_epatches \
#   --griffin \
#   --allow_code_execution \
#   --spr 0.5 \
#   --widthtree $treesize \
#   --check \
#   --kernelsize 16 \
#   --thr 0.1 \

# accelerate launch --num_processes 8 main.py \
#   --model meta-llama/Llama-2-7b-chat-hf \
#   --tasks humaneval \
#   --do_sample False \
#   --n_samples 1 \
#   --batch_size 1 \
#   --max_length_generation 512 \
#   --enable_epatches \
#   --griffin \
#   --allow_code_execution \
#   --spr 0.5 \
#   --widthtree $treesize \
#   --check \
#   --kernelsize 12 \
#   --thr 0.1 \

# done 

treesizes=(1 4 6 8)
for treesize in ${treesizes[@]} 
do 
# accelerate launch --num_processes 8 main.py \
#   --model meta-llama/Llama-2-7b-chat-hf \
#   --tasks humaneval \
#   --do_sample False \
#   --n_samples 1 \
#   --batch_size 1 \
#   --max_length_generation 512 \
#   --enable_epatches \
#   --griffin \
#   --allow_code_execution \
#   --spr 0.5 \
#   --widthtree $treesize \
#   --check \
#   --kernelsize 16 \
#   --thr 0.05 \

# accelerate launch --num_processes 4 main.py \
#   --model meta-llama/Llama-2-13b-hf \
#   --tasks mbppplus \
#   --do_sample False \
#   --n_samples 1 \
#   --batch_size 1 \
#   --max_length_generation 512 \
#   --enable_epatches \
#   --allow_code_execution \
#   --spr 0.5 \
#   --kernelsize 12 \
#   --limit 100 \

accelerate launch --num_processes 8 main.py \
  --model meta-llama/Meta-Llama-3-8B \
  --tasks humaneval \
  --do_sample False \
  --n_samples 1 \
  --batch_size 1 \
  --max_length_generation 512 \
  --enable_epatches \
  --cats \
  --allow_code_execution \
  --spr 0.5 \
  --widthtree $treesize \
  --check \
  --kernelsize 16 \
  --thr 0.05 \
  --patternstrict \

# accelerate launch --num_processes 3 main.py \
#   --model meta-llama/Meta-Llama-3-8B-Instruct \
#   --tasks mbppplus \
#   --do_sample False \
#   --n_samples 1 \
#   --batch_size 1 \
#   --max_length_generation 512 \
#   --enable_epatches \
#   --griffin \
#   --allow_code_execution \
#   --spr 0.5 \
#   --widthtree $treesize \
#   --check \
#   --kernelsize 16 \
#   --thr 0.1 \

done 

# treesizes=(1 4 6 8) 
# for treesize in ${treesizes[@]} 
# do 
# accelerate launch --num_processes 8 main.py \
#   --model meta-llama/Meta-Llama-3-8B \
#   --tasks humaneval \
#   --do_sample False \
#   --n_samples 1 \
#   --batch_size 1 \
#   --max_length_generation 512 \
#   --enable_epatches \
#   --griffin \
#   --allow_code_execution \
#   --spr 0.5 \
#   --widthtree $treesize \
#   --check \
#   --kernelsize 16 \
#   --thr 0.05 \

# done 

# accelerate launch --num_processes 4 main.py \
#   --model meta-llama/Meta-Llama-3-8B \
#   --limit 50 \
#   --tasks mbpp \
#   --do_sample False \
#   --n_samples 1 \
#   --batch_size 1 \
#   --max_length_generation 2048 \
#   --allow_code_execution