# vllm serve google/gemma-3-4b-it --host 0.0.0.0 --gpu_memory_utilization 0.9 --enforce-eager --max-model-len 1024 --cpu-offload-gb 4  --limit-mm-per-prompt image=1

vllm serve Qwen/Qwen2.5-VL-3B-Instruct-AWQ --host 0.0.0.0 --gpu_memory_utilization 0.9 --enforce-eager --max-model-len 8192 --cpu-offload-gb 0  --limit-mm-per-prompt image=1