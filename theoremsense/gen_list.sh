## Usage: bash gen.sh <model_name> <device>
#models = (
#  "deepseek-ai/deepseek-math-7b-instruct"
#  "deepseek-ai/deepseek-math-7b-rl"
#  "EleutherAI/llemma_7b"
#  "llm-agents/tora-13b-v1.0"
#  "llm-agents/tora-7b-v1.0"
#  "google/gemma-7b"
#  "morph-labs/morph-prover-v0-7b"
#  "lmsys/vicuna-13b-v1.5"
#  "lmsys/vicuna-7b-v1.5"
#  "mistralai/Mistral-7B-Instruct-v0.2"
#  "meta-llama/Llama-2-13b-chat-hf"
#  "meta-llama/Llama-2-7b-chat-hf"
#)

models=(
  "deepseek-ai/deepseek-math-7b-instruct"
  "deepseek-ai/deepseek-math-7b-rl"
  "EleutherAI/llemma_7b"
  "llm-agents/tora-13b-v1.0"
  "llm-agents/tora-7b-v1.0"
  "google/gemma-7b"
  "morph-labs/morph-prover-v0-7b"
  "lmsys/vicuna-13b-v1.5"
  "lmsys/vicuna-7b-v1.5"
  "mistralai/Mistral-7B-Instruct-v0.2"
  "meta-llama/Llama-2-13b-chat-hf"
  "meta-llama/Llama-2-7b-chat-hf"
)

device=6

for model in "${models[@]}";
do
  bash gen.sh $model $device
done