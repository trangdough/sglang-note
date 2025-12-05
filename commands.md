# Docker
## Start Container
```bash
# B200
docker run \
  -itd \
  --shm-size 32g \
  --gpus all \
  -v /root/.cache:/root/.cache \
  --ipc=host \
  --network=host \
  --privileged \
  --pid=host \
  --name sgl_trang \
  lmsysorg/sglang:blackwell \
  /bin/zsh
  
# H100
docker run \
  -itd \
  --shm-size 32g \
  --gpus all \
  -v /root/.cache:/root/.cache \
  --ipc=host \
  --network=host \
  --privileged \
  --pid=host \
  --name sgl_trang \
  lmsysorg/sglang:dev \
  /bin/zsh
  ```

## Enter Container
```bash
docker exec -it sgl_trang /bin/zsh
```

# Start a server
```bash
SGLANG_TORCH_PROFILER_DIR="/sgl-workspace/profile-traces" \
CUDA_VISIBLE_DEVICES="1"

# disable cuda graph
CUDA_VISIBLE_DEVICES="0" \
python3 -m sglang.launch_server \
  --model openai/gpt-oss-20b \
  --tp 1 \
  --port 30010 \
  --prefill-attention-backend fa4 \
  --page-size 128 \
  --decode-attention-backend triton \
  --disable-cuda-graph
  
CUDA_VISIBLE_DEVICES="0" python3 -m sglang.launch_server \
  --model openai/gpt-oss-20b --tp 1 --port 30010

CUDA_VISIBLE_DEVICES="0" python3 -m sglang.launch_server --model openai/gpt-oss-20b --tp 1 --port 30010 --prefill-attention-backend fa4
CUDA_VISIBLE_DEVICES="1" python3 -m sglang.launch_server --model openai/gpt-oss-20b --tp 1 --port 30010
```

# Send a query
```bash
curl -X POST http://localhost:30010/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "The Transformer architecture is", "sampling_params": {"temperature": 0}}' \
  | jq .text
  
  
# reference output:
" a type of neural network that was introduced in the paper \"Attention is All You Need\" by Vaswani et al. in 2017. It is a powerful and flexible architecture that has been used in a wide range of natural language processing tasks, including machine translation, language modeling, and text classification.\n\nThe Transformer architecture is based on the idea of self-attention, which allows the model to focus on different parts of the input sequence when making predictions. This is in contrast to traditional recurrent neural networks (RNNs), which process the input sequence one element at a time and can only attend to a fixed number of previous elements.\n\nThe"
```