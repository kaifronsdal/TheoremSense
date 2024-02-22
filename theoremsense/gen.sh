# Usage: bash gen.sh
# loop over models
models = (
  "deepseek-ai/deepseek-math-7b-instruct"
  "deepseek-ai/deepseek-math-7b-rl"
  "EleutherAI/llemma_7b"
  "llm-agents/tora-13b-v1.0"
  "microsoft/phi-2"
)

for model in "${models[@]}"
do
  python theoremsense/generate_predictions.py --model=$model --dataset=EleutherAI/hendrycks_math --subset=algebra --output=~/generated_outputs --perplexity --grade --num_shots=3 --load --override
done

python theoremsense/generate_predictions.py --model=deepseek-ai/deepseek-math-7b-instruct --dataset=EleutherAI/hendrycks_math --subset=algebra --output=~/generated_outputs --perplexity --grade --num_shots=3 --load --override

python theoremsense/generate_predictions.py --model=deepseek-ai/deepseek-math-7b-instruct --dataset=EleutherAI/hendrycks_math --output=~/generated_outputs --perplexity --grade --num_shots=3 --load --override

python theoremsense/generate_predictions.py --model=microsoft/phi-2 --dataset=EleutherAI/hendrycks_math --output=~/generated_outputs --perplexity --grade --num_shots=3 --load --override

python theoremsense/compute_TFA.py --model=deepseek-ai/deepseek-math-7b-instruct --dataset=EleutherAI/hendrycks_math --output=~/generated_outputs --tfa --grade --num_shots=3 --load --override