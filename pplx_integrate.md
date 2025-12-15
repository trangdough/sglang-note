# Architecture Diagram
```
┌─────────────────────────────────────────────────────────────┐
│                     User Model Layer                        │
│                  (DeepSeekV2MoE, Llama4MoE, etc.)           │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    get_moe_impl_class()                     │
│            Selects: FusedMoE / EPMoE / DeepEPMoE / PPLXMoE  │
└─────────────────────────┬───────────────────────────────────┘
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
    ┌─────────┐     ┌──────────┐     ┌──────────┐
    │ DeepEP  │     │  PPLX    │     │   None   │
    │ Backend │     │ Backend  │     │ (Local)  │
    └────┬────┘     └────┬─────┘     └────┬─────┘
         │               │                │
         ▼               ▼                ▼
    ┌─────────────────────────────────────────┐
    │           sgl_kernel (C++/CUDA)         │
    │  • all-to-all ops  • grouped GEMM       │
    │  • dispatch/combine • quantization      │
    └─────────────────────────────────────────┘
```
I'll create a detailed step-by-step implementation guide with flowcharts showing both the data flow and implementation order.

---

# PPLX Integration - Detailed Implementation Guide

## Implementation Flowchart

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           IMPLEMENTATION ORDER FLOWCHART                                 │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: PREREQUISITES                                                                   │
│ ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                       │
│ │ Install NVSHMEM │───▶│ Install PPLX    │───▶│ Verify imports  │                       │
│ │ Library         │    │ Kernels         │    │ work            │                       │
│ └─────────────────┘    └─────────────────┘    └─────────────────┘                       │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: ADD ENUMS (base.py, utils.py)                                                   │
│ ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                       │
│ │DispatchOutput   │───▶│CombineInput     │───▶│MoeA2ABackend    │                       │
│ │Format.PPLX      │    │Format.PPLX      │    │.PPLX            │                       │
│ └─────────────────┘    └─────────────────┘    └─────────────────┘                       │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: ADD CLI OPTION (server_args.py)                                                 │
│ ┌─────────────────────────────────────────────────────────────────────────────────────┐ │
│ │ MOE_A2A_BACKEND_CHOICES = ["none", "deepep", "mooncake", "ascend_fuseep", "pplx"]   │ │
│ └─────────────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: CREATE DISPATCHER (pplx.py) - NEW FILE                                          │
│ ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                       │
│ │PPLXDispatch     │───▶│PPLXCombine      │───▶│PPLXDispatcher   │                       │
│ │Output           │    │Input            │    │(BaseDispatcher) │                       │
│ └─────────────────┘    └─────────────────┘    └─────────────────┘                       │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ STEP 5: EXPORT CLASSES (__init__.py)                                                    │
│ ┌─────────────────────────────────────────────────────────────────────────────────────┐ │
│ │ from .pplx import PPLXDispatcher, PPLXDispatchOutput, PPLXCombineInput              │ │
│ └─────────────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ STEP 6: WIRE UP FACTORY (fused_moe_triton/layer.py)                                     │
│ ┌─────────────────────────────────────────────────────────────────────────────────────┐ │
│ │ create_moe_dispatcher() ──▶ if a2a_backend.is_pplx(): return PPLXDispatcher(...)    │ │
│ └─────────────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ STEP 7: TEST & VERIFY                                                                   │
│ ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                       │
│ │ Unit Tests      │───▶│ Integration     │───▶│ E2E Model Test  │                       │
│ │                 │    │ Tests           │    │                 │                       │
│ └─────────────────┘    └─────────────────┘    └─────────────────┘                       │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Flowchart

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              RUNTIME DATA FLOW                                           │
└─────────────────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────────────────────────┐
    │                              INPUT TENSORS                                        │
    │  hidden_states: [num_tokens, hidden_dim]  topk_ids: [num_tokens, top_k]          │
    │  topk_weights: [num_tokens, top_k]                                               │
    └──────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
    ┌──────────────────────────────────────────────────────────────────────────────────┐
    │                         PPLXDispatcher.dispatch()                                 │
    │  ┌────────────────────────────────────────────────────────────────────────────┐  │
    │  │ 1. Convert topk_ids to uint32 (PPLX requirement)                           │  │
    │  │ 2. Allocate output buffers:                                                │  │
    │  │    • expert_num_tokens: [num_local_experts]                                │  │
    │  │    • expert_x: [num_local_experts, max_recv_tokens, hidden_dim]            │  │
    │  │ 3. Call AllToAll.dispatch()                                                │  │
    │  └────────────────────────────────────────────────────────────────────────────┘  │
    └──────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
    ┌──────────────────────────────────────────────────────────────────────────────────┐
    │                          AllToAll.dispatch()                                      │
    │  ┌────────────────────────────────────────────────────────────────────────────┐  │
    │  │                         NVSHMEM All-to-All                                 │  │
    │  │  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐                  │  │
    │  │  │ GPU 0   │◄──►│ GPU 1   │◄──►│ GPU 2   │◄──►│ GPU 3   │                  │  │
    │  │  │ Expert  │    │ Expert  │    │ Expert  │    │ Expert  │                  │  │
    │  │  │ 0,1     │    │ 2,3     │    │ 4,5     │    │ 6,7     │                  │  │
    │  │  └─────────┘    └─────────┘    └─────────┘    └─────────┘                  │  │
    │  │  Tokens routed to experts 0,1 → GPU 0                                      │  │
    │  │  Tokens routed to experts 2,3 → GPU 1                                      │  │
    │  │  ...                                                                       │  │
    │  └────────────────────────────────────────────────────────────────────────────┘  │
    └──────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
    ┌──────────────────────────────────────────────────────────────────────────────────┐
    │                          PPLXDispatchOutput                                       │
    │  • hidden_states: [num_local_experts, max_recv, hidden_dim] (tokens per expert)  │
    │  • expert_num_tokens: [num_local_experts] (count of tokens per expert)           │
    │  • topk_ids, topk_weights (preserved for combine)                                │
    └──────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
    ┌──────────────────────────────────────────────────────────────────────────────────┐
    │                          MoeRunner.forward()                                      │
    │  ┌────────────────────────────────────────────────────────────────────────────┐  │
    │  │ For each local expert i with num_tokens > 0:                               │  │
    │  │   expert_input = hidden_states[i, :num_tokens]                             │  │
    │  │   expert_output = W2(activation(W1(expert_input)))                         │  │
    │  └────────────────────────────────────────────────────────────────────────────┘  │
    └──────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
    ┌──────────────────────────────────────────────────────────────────────────────────┐
    │                           PPLXCombineInput                                        │
    │  • hidden_states: expert outputs (same shape as dispatch output)                 │
    │  • topk_ids, topk_weights, num_input_tokens                                      │
    └──────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
    ┌──────────────────────────────────────────────────────────────────────────────────┐
    │                         PPLXDispatcher.combine()                                  │
    │  ┌────────────────────────────────────────────────────────────────────────────┐  │
    │  │ 1. Allocate output: out_tokens [max_num_tokens, hidden_dim]                │  │
    │  │ 2. Call AllToAll.combine()                                                 │  │
    │  │ 3. Return out_tokens[:num_input_tokens]                                    │  │
    │  └────────────────────────────────────────────────────────────────────────────┘  │
    └──────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
    ┌──────────────────────────────────────────────────────────────────────────────────┐
    │                          AllToAll.combine()                                       │
    │  ┌────────────────────────────────────────────────────────────────────────────┐  │
    │  │                    NVSHMEM All-to-All (Reverse)                            │  │
    │  │  • Gather outputs from all GPUs                                            │  │
    │  │  • Weighted sum: output[i] = Σ(weight[i,k] * expert_output[k])             │  │
    │  └────────────────────────────────────────────────────────────────────────────┘  │
    └──────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
    ┌──────────────────────────────────────────────────────────────────────────────────┐
    │                              OUTPUT TENSOR                                        │
    │  final_hidden_states: [num_tokens, hidden_dim]                                   │
    └──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Step-by-Step Implementation

### STEP 1: Prerequisites

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ STEP 1A: Install NVSHMEM                                                                │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ Commands:                                                                               │
│   wget https://developer.nvidia.com/downloads/assets/secure/nvshmem/nvshmem_src_3.2.5-1.txz
│   tar xf nvshmem_src_3.2.5-1.txz                                                        │
│   cd nvshmem_src_3.2.5-1/nvshmem_src && mkdir build && cd build                         │
│   cmake -DNVSHMEM_PREFIX=/opt/nvshmem-3.2.5 -DCMAKE_CUDA_ARCHITECTURES=90a -G Ninja ..  │
│   ninja && sudo ninja install                                                           │
│                                                                                         │
│ Environment:                                                                            │
│   export NVSHMEM_HOME=/opt/nvshmem-3.2.5                                                │
│   export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH                             │
│   export NVSHMEM_REMOTE_TRANSPORT=none  # single-node                                   │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ STEP 1B: Install PPLX Kernels                                                           │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ Commands:                                                                               │
│   cd /root/pplx-kernels                                                                 │
│   TORCH_CUDA_ARCH_LIST=9.0a+PTX python3 setup.py bdist_wheel                            │
│   pip install dist/*.whl                                                                │
│                                                                                         │
│ Verify:                                                                                 │
│   python3 -c "from pplx_kernels import AllToAll; print('OK')"                           │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### STEP 2: Add Format Enums

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ STEP 2A: Modify base.py - Add DispatchOutputFormat.PPLX                                 │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ File: sglang/python/sglang/srt/layers/moe/token_dispatcher/base.py                      │
│ Location: Line ~153 (DispatchOutputFormat class)                                        │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ BEFORE:                                                                                 │
│ ┌─────────────────────────────────────────────────────────────────────────────────────┐ │
│ │ class DispatchOutputFormat(Enum):                                                   │ │
│ │     STANDARD = "standard"                                                           │ │
│ │     DEEPEP_NORMAL = "deepep_normal"                                                 │ │
│ │     DEEPEP_LL = "deepep_ll"                                                         │ │
│ └─────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                         │
│ AFTER:                                                                                  │
│ ┌─────────────────────────────────────────────────────────────────────────────────────┐ │
│ │ class DispatchOutputFormat(Enum):                                                   │ │
│ │     STANDARD = "standard"                                                           │ │
│ │     DEEPEP_NORMAL = "deepep_normal"                                                 │ │
│ │     DEEPEP_LL = "deepep_ll"                                                         │ │
│ │     PPLX = "pplx"  # ← ADD THIS                                                     │ │
│ │                                                                                     │ │
│ │     def is_pplx(self) -> bool:  # ← ADD THIS METHOD                                 │ │
│ │         return self == DispatchOutputFormat.PPLX                                    │ │
│ └─────────────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ STEP 2B: Modify base.py - Add CombineInputFormat.PPLX                                   │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ File: sglang/python/sglang/srt/layers/moe/token_dispatcher/base.py                      │
│ Location: Line ~217 (CombineInputFormat class)                                          │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ BEFORE:                                                                                 │
│ ┌─────────────────────────────────────────────────────────────────────────────────────┐ │
│ │ class CombineInputFormat(Enum):                                                     │ │
│ │     STANDARD = "standard"                                                           │ │
│ │     DEEPEP_NORMAL = "deepep_normal"                                                 │ │
│ │     DEEPEP_LL = "deepep_ll"                                                         │ │
│ └─────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                         │
│ AFTER:                                                                                  │
│ ┌─────────────────────────────────────────────────────────────────────────────────────┐ │
│ │ class CombineInputFormat(Enum):                                                     │ │
│ │     STANDARD = "standard"                                                           │ │
│ │     DEEPEP_NORMAL = "deepep_normal"                                                 │ │
│ │     DEEPEP_LL = "deepep_ll"                                                         │ │
│ │     PPLX = "pplx"  # ← ADD THIS                                                     │ │
│ └─────────────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ STEP 2C: Modify utils.py - Add MoeA2ABackend.PPLX                                       │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ File: sglang/python/sglang/srt/layers/moe/utils.py                                      │
│ Location: Line ~22 (MoeA2ABackend class)                                                │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ BEFORE:                                                                                 │
│ ┌─────────────────────────────────────────────────────────────────────────────────────┐ │
│ │ class MoeA2ABackend(Enum):                                                          │ │
│ │     NONE = "none"                                                                   │ │
│ │     DEEPEP = "deepep"                                                               │ │
│ │     MOONCAKE = "mooncake"                                                           │ │
│ │     ASCEND_FUSEEP = "ascend_fuseep"                                                 │ │
│ └─────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                         │
│ AFTER:                                                                                  │
│ ┌─────────────────────────────────────────────────────────────────────────────────────┐ │
│ │ class MoeA2ABackend(Enum):                                                          │ │
│ │     NONE = "none"                                                                   │ │
│ │     DEEPEP = "deepep"                                                               │ │
│ │     MOONCAKE = "mooncake"                                                           │ │
│ │     ASCEND_FUSEEP = "ascend_fuseep"                                                 │ │
│ │     PPLX = "pplx"  # ← ADD THIS                                                     │ │
│ │                                                                                     │ │
│ │     # ... existing methods ...                                                      │ │
│ │                                                                                     │ │
│ │     def is_pplx(self):  # ← ADD THIS METHOD                                         │ │
│ │         return self == MoeA2ABackend.PPLX                                           │ │
│ └─────────────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### STEP 3: Add CLI Option

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: Modify server_args.py                                                           │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ File: sglang/python/sglang/srt/server_args.py                                           │
│ Location: Line ~173 (MOE_A2A_BACKEND_CHOICES)                                           │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ BEFORE:                                                                                 │
│ ┌─────────────────────────────────────────────────────────────────────────────────────┐ │
│ │ MOE_A2A_BACKEND_CHOICES = ["none", "deepep", "mooncake", "ascend_fuseep"]           │ │
│ └─────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                         │
│ AFTER:                                                                                  │
│ ┌─────────────────────────────────────────────────────────────────────────────────────┐ │
│ │ MOE_A2A_BACKEND_CHOICES = ["none", "deepep", "mooncake", "ascend_fuseep", "pplx"]   │ │
│ └─────────────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### STEP 4: Create PPLX Dispatcher (NEW FILE)

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: Create pplx.py                                                                  │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ File: sglang/python/sglang/srt/layers/moe/token_dispatcher/pplx.py (NEW)                │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ Structure:                                                                              │
│                                                                                         │
│ ┌─────────────────────────────────────────────────────────────────────────────────────┐ │
│ │ # SECTION 1: Imports                                                                │ │
│ │ from pplx_kernels import nvshmem_init                                               │ │
│ │ from pplx_kernels.all_to_all import AllToAll                                        │ │
│ │ from sglang.srt.layers.moe.token_dispatcher.base import BaseDispatcher, ...         │ │
│ └─────────────────────────────────────────────────────────────────────────────────────┘ │
│                                          │                                              │
│                                          ▼                                              │
│ ┌─────────────────────────────────────────────────────────────────────────────────────┐ │
│ │ # SECTION 2: NVSHMEM Initialization                                                 │ │
│ │ _nvshmem_initialized = False                                                        │ │
│ │                                                                                     │ │
│ │ def ensure_nvshmem_initialized(global_rank, local_rank, world_size):                │ │
│ │     global _nvshmem_initialized                                                     │ │
│ │     if _nvshmem_initialized: return                                                 │ │
│ │     dev = Device(local_rank)                                                        │ │
│ │     dev.set_current()                                                               │ │
│ │     nvshmem_init(global_rank, local_rank, world_size, dev)                          │ │
│ │     _nvshmem_initialized = True                                                     │ │
│ └─────────────────────────────────────────────────────────────────────────────────────┘ │
│                                          │                                              │
│                                          ▼                                              │
│ ┌─────────────────────────────────────────────────────────────────────────────────────┐ │
│ │ # SECTION 3: Output/Input Types                                                     │ │
│ │                                                                                     │ │
│ │ class PPLXDispatchOutput(NamedTuple):                                               │ │
│ │     hidden_states: torch.Tensor        # [num_local_experts, max_recv, hidden]      │ │
│ │     hidden_states_scale: Optional[...]  # FP8 scales                                │ │
│ │     topk_ids: torch.Tensor             # [num_tokens, top_k]                        │ │
│ │     topk_weights: torch.Tensor         # [num_tokens, top_k]                        │ │
│ │     expert_num_tokens: torch.Tensor    # [num_local_experts]                        │ │
│ │     num_input_tokens: int              # Original batch size                        │ │
│ │                                                                                     │ │
│ │     @property                                                                       │ │
│ │     def format(self) -> DispatchOutputFormat:                                       │ │
│ │         return DispatchOutputFormat.PPLX                                            │ │
│ │                                                                                     │ │
│ │ class PPLXCombineInput(NamedTuple):                                                 │ │
│ │     hidden_states: torch.Tensor        # Expert outputs                             │ │
│ │     topk_ids: torch.Tensor                                                          │ │
│ │     topk_weights: torch.Tensor                                                      │ │
│ │     num_input_tokens: int                                                           │ │
│ │                                                                                     │ │
│ │     @property                                                                       │ │
│ │     def format(self) -> CombineInputFormat:                                         │ │
│ │         return CombineInputFormat.PPLX                                              │ │
│ └─────────────────────────────────────────────────────────────────────────────────────┘ │
│                                          │                                              │
│                                          ▼                                              │
│ ┌─────────────────────────────────────────────────────────────────────────────────────┐ │
│ │ # SECTION 4: PPLXDispatcher Class                                                   │ │
│ │                                                                                     │ │
│ │ class PPLXDispatcher(BaseDispatcher):                                               │ │
│ │                                                                                     │ │
│ │     def __init__(self, group, router_topk, num_experts, ...):                       │ │
│ │         # 1. Store config                                                           │ │
│ │         # 2. Calculate dp_size = world_size // (num_experts // num_local_experts)   │ │
│ │         # 3. Call ensure_nvshmem_initialized()                                      │ │
│ │         # 4. Create AllToAll instance                                               │ │
│ │         self.ata = AllToAll.intranode(...) or AllToAll.internode(...)               │ │
│ │                                                                                     │ │
│ │     def dispatch(self, hidden_states, topk_output) -> PPLXDispatchOutput:           │ │
│ │         # 1. Convert topk_ids to uint32                                             │ │
│ │         # 2. Allocate output buffers                                                │ │
│ │         # 3. Call self.ata.dispatch(...)                                            │ │
│ │         # 4. Return PPLXDispatchOutput(...)                                         │ │
│ │                                                                                     │ │
│ │     def combine(self, combine_input) -> torch.Tensor:                               │ │
│ │         # 1. Allocate output buffer                                                 │ │
│ │         # 2. Call self.ata.combine(...)                                             │ │
│ │         # 3. Return out_tokens[:num_input_tokens]                                   │ │
│ │                                                                                     │ │
│ │     def destroy(self):                                                              │ │
│ │         self.ata.destroy()                                                          │ │
│ └─────────────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### STEP 5: Export Classes

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ STEP 5: Modify __init__.py                                                              │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ File: sglang/python/sglang/srt/layers/moe/token_dispatcher/__init__.py                  │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ ADD these imports:                                                                      │
│ ┌─────────────────────────────────────────────────────────────────────────────────────┐ │
│ │ from sglang.srt.layers.moe.token_dispatcher.pplx import (                           │ │
│ │     PPLXCombineInput,                                                               │ │
│ │     PPLXDispatcher,                                                                 │ │
│ │     PPLXDispatchOutput,                                                             │ │
│ │ )                                                                                   │ │
│ └─────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                         │
│ ADD to __all__ list:                                                                    │
│ ┌─────────────────────────────────────────────────────────────────────────────────────┐ │
│ │ __all__ = [                                                                         │ │
│ │     # ... existing exports ...                                                      │ │
│ │     "PPLXDispatcher",                                                               │ │
│ │     "PPLXDispatchOutput",                                                           │ │
│ │     "PPLXCombineInput",                                                             │ │
│ │ ]                                                                                   │ │
│ └─────────────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### STEP 6: Wire Up Factory

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ STEP 6: Modify fused_moe_triton/layer.py                                                │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ File: sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py                     │
│ Location: Line ~79 (create_moe_dispatcher function)                                     │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ BEFORE:                                                                                 │
│ ┌─────────────────────────────────────────────────────────────────────────────────────┐ │
│ │ def create_moe_dispatcher(moe_runner_config: MoeRunnerConfig) -> BaseDispatcher:    │ │
│ │     a2a_backend = get_moe_a2a_backend()                                             │ │
│ │     if a2a_backend.is_none():                                                       │ │
│ │         return StandardDispatcher(moe_runner_config)                                │ │
│ │     elif a2a_backend.is_deepep() or a2a_backend.is_mooncake():                      │ │
│ │         return MaybeTboDeepEPDispatcher(...)                                        │ │
│ │     elif a2a_backend.is_ascend_fuseep():                                            │ │
│ │         return NpuFuseEPDispatcher(...)                                             │ │
│ │     else:                                                                           │ │
│ │         raise NotImplementedError(...)                                              │ │
│ └─────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                         │
│ AFTER:                                                                                  │
│ ┌─────────────────────────────────────────────────────────────────────────────────────┐ │
│ │ def create_moe_dispatcher(moe_runner_config: MoeRunnerConfig) -> BaseDispatcher:    │ │
│ │     a2a_backend = get_moe_a2a_backend()                                             │ │
│ │     if a2a_backend.is_none():                                                       │ │
│ │         return StandardDispatcher(moe_runner_config)                                │ │
│ │     elif a2a_backend.is_deepep() or a2a_backend.is_mooncake():                      │ │
│ │         return MaybeTboDeepEPDispatcher(...)                                        │ │
│ │                                                                                     │ │
│ │     elif a2a_backend.is_pplx():  # ← ADD THIS BLOCK                                 │ │
│ │         from sglang.srt.layers.moe.token_dispatcher.pplx import PPLXDispatcher      │ │
│ │                                                                                     │ │
│ │         return PPLXDispatcher(                                                      │ │
│ │             group=get_tp_group().device_group,                                      │ │
│ │             router_topk=moe_runner_config.top_k,                                    │ │
│ │             num_experts=moe_runner_config.num_experts,                              │ │
│ │             num_local_experts=moe_runner_config.num_local_experts,                  │ │
│ │             hidden_size=moe_runner_config.hidden_size,                              │ │
│ │             max_num_tokens=1024,  # or from config                                  │ │
│ │             params_dtype=moe_runner_config.params_dtype,                            │ │
│ │             internode=False,                                                        │ │
│ │         )                                                                           │ │
│ │                                                                                     │ │
│ │     elif a2a_backend.is_ascend_fuseep():                                            │ │
│ │         return NpuFuseEPDispatcher(...)                                             │ │
│ │     else:                                                                           │ │
│ │         raise NotImplementedError(...)                                              │ │
│ └─────────────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### STEP 7: Test & Verify

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ STEP 7A: Unit Test                                                                      │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ Command:                                                                                │
│   pytest -xvs test/srt/ep/test_pplx.py                                                  │
│                                                                                         │
│ Test covers:                                                                            │
│   ✓ PPLXDispatcher initialization                                                       │
│   ✓ dispatch() returns correct output shape                                             │
│   ✓ combine() returns correct output shape                                              │
│   ✓ Resource cleanup via destroy()                                                      │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ STEP 7B: Integration Test                                                               │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ Command:                                                                                │
│   torchrun --nproc_per_node=4 test/manual/ep/test_pplx_moe.py                           │
│                                                                                         │
│ Test covers:                                                                            │
│   ✓ Multi-GPU dispatch/combine flow                                                     │
│   ✓ Token routing correctness                                                           │
│   ✓ NVSHMEM initialization                                                              │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ STEP 7C: End-to-End Model Test                                                          │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ Command:                                                                                │
│   python -m sglang.launch_server \                                                      │
│       --model-path Qwen/Qwen2-MoE-57B-A14B \                                            │
│       --moe-a2a-backend pplx \                                                          │
│       --tp 4 --ep 4 \                                                                   │
│       --port 30000                                                                      │
│                                                                                         │
│ Verify:                                                                                 │
│   curl http://localhost:30000/generate -d '{"text": "Hello", "max_tokens": 10}'         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## File Dependency Graph

```
                    ┌─────────────────────────────────────┐
                    │           server_args.py             │
                    │  MOE_A2A_BACKEND_CHOICES += "pplx"   │
                    └─────────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │             utils.py                 │
                    │    MoeA2ABackend.PPLX + is_pplx()    │
                    └─────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
    ┌───────────────────────────┐       ┌───────────────────────────┐
    │         base.py            │       │   fused_moe_triton/       │
    │  DispatchOutputFormat.PPLX │       │        layer.py           │
    │  CombineInputFormat.PPLX   │       │  create_moe_dispatcher()  │
    └───────────────────────────┘       └───────────────────────────┘
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      ▼
                    ┌─────────────────────────────────────┐
                    │             pplx.py (NEW)            │
                    │  PPLXDispatchOutput                  │
                    │  PPLXCombineInput                    │
                    │  PPLXDispatcher                      │
                    └─────────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │           __init__.py                │
                    │  Export: PPLXDispatcher, etc.        │
                    └─────────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │       pplx_kernels (external)        │
                    │  AllToAll.intranode()                │
                    │  AllToAll.internode()                │
                    │  nvshmem_init()                      │
                    └─────────────────────────────────────┘
```

---

## Implementation Checklist

| Step | File | Change | Done |
|------|------|--------|------|
| 1A | System | Install NVSHMEM | ⬜ |
| 1B | System | Install PPLX kernels | ⬜ |
| 2A | `base.py` | Add `DispatchOutputFormat.PPLX` | ⬜ |
| 2B | `base.py` | Add `CombineInputFormat.PPLX` | ⬜ |
| 2C | `utils.py` | Add `MoeA2ABackend.PPLX` | ⬜ |
| 3 | `server_args.py` | Add `"pplx"` to choices | ⬜ |
| 4 | `pplx.py` | Create new dispatcher file | ⬜ |
| 5 | `__init__.py` | Export PPLX classes | ⬜ |
| 6 | `layer.py` | Add PPLX to factory | ⬜ |
| 7A | Test | Unit tests | ⬜ |
| 7B | Test | Integration tests | ⬜ |
| 7C | Test | E2E model test | ⬜ |
