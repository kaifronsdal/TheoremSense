## Usage: bash gen.sh <model_name> <device>
#models = (
#  "deepseek-ai/deepseek-math-7b-instruct"
#  "deepseek-ai/deepseek-math-7b-rl"
#  "EleutherAI/llemma_7b"
#  "llm-agents/tora-13b-v1.0"
#  "microsoft/phi-2"
#)

model_name=$1
dataset="EleutherAI/hendrycks_math"
output_path="~/generated_outputs"
num_shots=3
batch_size=4
override=True

# set CUDA_VISIBLE_DEVICES
export PCI_BUS_ID="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=$2
#export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

python generate_predictions.py --model=$model_name --dataset=$dataset --output=$output_path \
  --num_shots=$num_shots --batch_size=$batch_size --override=$override --method=autoregressive

python generate_predictions.py --model=$model_name --dataset=$dataset --output=$output_path \
  --num_shots=$num_shots --batch_size=$batch_size --override=$override --method=teacher_forcing