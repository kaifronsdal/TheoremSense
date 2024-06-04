# args.model = 'mistralai/Mistral-7B-Instruct-v0.2'
# args.use_chat = True
# args.model = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
# args.use_chat = True
# args.model = 'tiiuae/falcon-7b'  # Don't use?
# args.use_chat = False
# args.model = 'meta-llama/Llama-2-7b-chat-hf'
# args.use_chat = True
# args.model = 'meta-llama/Llama-2-13b-chat-hf'
# args.use_chat = True
# args.model = 'meta-llama/Llama-2-70b-chat-hf'
# args.use_chat = True
# args.model = 'meta-llama/Meta-Llama-3-8B-Instruct'
# args.use_chat = True
# args.model = 'meta-llama/Meta-Llama-3-70B-Instruct'
# args.use_chat = True
# args.model = 'deepseek-ai/deepseek-math-7b-instruct'
# args.use_chat = True
# args.model = 'deepseek-ai/deepseek-llm-67b-chat'
# args.use_chat = True
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=5,6
# loop over all models, num_gpu pairs
args=(
#    "meta-llama/Meta-Llama-3-70B-Instruct" 4
#    "mistralai/Mixtral-8x7B-Instruct-v0.1" 4
#    "meta-llama/Llama-2-70b-chat-hf" 4
#    "deepseek-ai/deepseek-llm-67b-chat" 4
#

#    "Qwen/Qwen-72B" 4
#    "Qwen/Qwen-14B" 1
#    "Qwen/Qwen-7B" 1
#
#    "mistralai/Mistral-7B-Instruct-v0.2" 1
#    "meta-llama/Llama-2-7b-chat-hf" 1
#    "meta-llama/Llama-2-13b-chat-hf" 1
#    "meta-llama/Meta-Llama-3-8B-Instruct" 1
#    "deepseek-ai/deepseek-math-7b-instruct" 1
#    "deepseek-ai/deepseek-math-7b-rl" 1

#    "EleutherAI/llemma_7b" 2
#    "GAIR/Abel-7B-002" 2
#    "google/gemma-7b-it" 2
#    "google/gemma-1.1-7b-it" 2

#    "EleutherAI/llemma_34b" 4

    "mistralai/Mixtral-8x22B-v0.1" 4
    "mistralai/Mixtral-8x22B-Instruct-v0.1" 4
)

for ((i=0; i<${#args[@]}; i+=2))
do
    model=${args[i]}
    num_gpu=${args[i+1]}

#    echo "python model_generate.py --num_shots 4 --output ~/model_evals --dataset math --detect_chat --model $model --tensor_parallel_size $num_gpu --backend hf --method teacher_forcing --batch_size=8 --split test"
#    python model_generate.py --num_shots 4 --output ~/model_evals --dataset math --detect_chat --model $model --tensor_parallel_size $num_gpu --backend hf --method teacher_forcing --batch_size=8 --split test
    echo "python model_generate.py --num_shots 4 --output ~/model_evals --dataset math --detect_chat --model $model --tensor_parallel_size $num_gpu --split train"
    python model_generate.py --num_shots 4 --output ~/model_evals --dataset math --detect_chat --model $model --tensor_parallel_size $num_gpu --split train
done
