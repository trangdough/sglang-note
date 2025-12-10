# MoeRunner Data Flow

┌─────────────────────────────────────────────────────────────────┐
│ Dispatcher Output                                               │
│   - hidden_states: [num_tokens, hidden_dim]                     │
│   - topk_ids: [num_tokens, top_k]  (which experts each token    │
│                                     should go to)               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ PRE_PERMUTE (compute indices, NO data movement)                 │
│                                                                 │
│   sorted_token_ids = moe_align_block_size(topk_ids, ...)        │
│                                                                 │
│   This creates an index array like:                             │
│   [token_3, token_7, token_1, ...]  ← "Expert 0 should process  │
│                                        these tokens"            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ KERNEL (actual permutation via gather/scatter)                  │
│                                                                 │
│   for each expert:                                              │
│       tokens = hidden_states[sorted_token_ids[expert_range]]    │
│       output = expert_weights @ tokens   ← GEMM                 │
│       results[sorted_token_ids[expert_range]] = output          │
│                                                                 │
│   The kernel uses indices to READ from scattered locations      │
│   and WRITE back to correct positions                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ POST_PERMUTE (format output for combiner)                       │
│   - Just wraps output tensor in expected data structure         │
└─────────────────────────────────────────────────────────────────┘
