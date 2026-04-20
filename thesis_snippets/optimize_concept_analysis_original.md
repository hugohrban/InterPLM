# Optimizing Concept–Feature Attribution: From Nested Sparse CPU Loops to Vectorized Dense GPU Accumulation

  Problem Setting

  For each shard of an evaluation set, we want to score every (concept, feature, threshold) triple against every amino-acid token. Let:

  - $T \approx 3 \times 10^5$ — tokens per shard
  - $F = 10^4$ — SAE features
  - $C = 5 \times 10^2$ — concepts
  - $K = 5$ — activation thresholds
  - $d = 320$ — PLM embedding dimension

  Given feature activations $A \in \mathbb{R}^{T \times F}$ (from the SAE encoder) and a label matrix $L \in \mathbb{Z}{\geq 0}^{T \times C}$ (where $L{t,c} =
  0$ means concept $c$ is absent at token $t$, and $L_{t,c} = d$ encodes domain instance $d$ otherwise), we compute three tensors of shape $(C, F, K)$:

  $$
  \mathrm{TP}{c,f,k} = \sum{t} \mathbf{1}[A_{t,f} > \tau_k] \cdot \mathbf{1}[L_{t,c} > 0]
  $$

  $$
  \mathrm{FP}{c,f,k} = \sum{t} \mathbf{1}[A_{t,f} > \tau_k] \cdot \mathbf{1}[L_{t,c} = 0]
  $$

  $$
  \mathrm{TP^{dom}}{c,f,k} = \bigl| { d > 0 : \exists, t,; L{t,c} = d \wedge A_{t,f} > \tau_k } \bigr|
  $$

  The first two count true/false positives at the token level; the third counts the number of distinct domain instances that feature $f$ "catches" for a
  non-AA-level concept $c$ (e.g. feature 42 fires somewhere in 17 separate zinc-finger instances).

  Original Algorithm

  The reference implementation had three nested loops surrounding sparse CPU matrix arithmetic:

  for feature_chunk in split(range(F), 40):           # F/250 = 40 chunks
      A_chunk = SAE.encode(embeddings)[:, feature_chunk]   # (T, 250)
      A_sparse = to_sparse(A_chunk.cpu().numpy())
      for k in range(K):                              # 5 thresholds
          B = (A_sparse.data > τ_k)                   # binarize
          for c in range(C):                          # 500 concepts
              TP[c, chunk, k] = (B ⊙ (L[:, c] > 0)).sum(axis=0)
              FP[c, chunk, k] = (B ⊙ (L[:, c] == 0)).sum(axis=0)
              if not aa_level[c]:
                  TP_dom[c, chunk, k] = count_unique(B ⊙ L[:, c])

  The cost decomposition per shard:

  (i) SAE forward passes: $F / 250 = 40$ feature chunks, each processing all $T$ tokens in sub-batches of 1024. Because encode_feat_subset just slices the
  encoder weight to the requested columns, the per-token FLOPs scale linearly with the chunk width, so the total SAE FLOPs are the same as one full-width pass
  — but the overhead (tensor allocation, normalization, torch.vstack of 300 sub-batches, GPU→CPU transfer, scipy.sparse.csr_matrix construction) is paid 40
  times.

  (ii) Metric computation — the real bottleneck: The innermost loop executes

  $$
  \text{iterations} = 40 \cdot K \cdot C = 40 \cdot 5 \cdot 500 = 100{,}000
  $$

  sparse element-wise multiplies and reductions on CPU. Each iteration allocates a new sparse matrix, broadcasts a (T, 1) column across (T, 250), and reduces
  along axis 0. The per-op cost is $\mathcal{O}(\mathrm{nnz})$ but the Python/SciPy overhead per call — CSR header construction, eliminate_zeros, attribute
  lookups, np.asarray(...).ravel() — dominates, because $\mathrm{nnz}$ per column-slice is small.

  (iii) Domain counting: count_unique_nonzero_sparse iterates over each of the 250 feature columns in a chunk, extracts col.data, and builds a Python set.
  Executed $40 \cdot K \cdot C_{\mathrm{non\text{-}aa}} \cdot 250 \approx 10^7$ times per shard — pure Python set operations.

  Optimization 1: Vectorizing the Concept Loop

  The per-concept loop for TP/FP can be written as a single sparse matrix product. Let $B_k \in {0,1}^{T \times F}$ be the threshold-binarized activations and
  $\tilde L \in {0,1}^{T \times C}$ be $\mathbf{1}[L > 0]$. Then

  $$
  \mathrm{TP}_{:,:,k} = \tilde L^\top B_k \quad\in\quad \mathbb{R}^{C \times F}
  $$

  and, since FP counts tokens where the feature fires but the concept is absent,

  $$
  \mathrm{FP}{:,:,k} = \mathbf{1}^\top B_k - \mathrm{TP}{:,:,k}.
  $$

  This collapses the $C = 500$ inner iterations into one matmul per threshold. Algorithmically it's the same work ($\mathcal{O}(\mathrm{nnz}(\tilde L) \cdot
  \bar F_{\mathrm{active}})$), but it's executed in one SciPy call instead of 500 Python-level ones — eliminating the Python overhead that dominated the
  original. This already gave a large speedup with minimal structural change.

  Optimization 2: Loop Inversion + Dense GPU Accumulation

  The deeper issue is that sparse CPU matmul on a $(C \times T)(T \times F)$ product is the wrong tool when $T \cdot F$ is only a few billion. The data isn't
  sparse enough for SpGEMM to beat dense BLAS — and certainly not dense GPU BLAS.

  We invert the loop order: instead of processing all $T$ tokens for 40 feature chunks, we process all $F$ features for $T / B$ token chunks (with $B = 4096$):

  $$
  \underbrace{\mathrm{TP}{:,:,k}}{(C,F)} ;\mathrel{+}=; \underbrace{\tilde L_{[b:b+B]}^\top}{(C, B)} \cdot \underbrace{B_k^{[b:b+B]}}{(B, F)}
  $$

  This is a dense matmul on the GPU with shapes $(500, 4096) \times (4096, 10^4)$, producing a $(500, 10^4)$ increment. The per-matmul FLOP count is

  $$
  2 \cdot C \cdot B \cdot F = 2 \cdot 500 \cdot 4096 \cdot 10^4 \approx 4 \times 10^{10}
  $$

  On an A100 (312 TF32 TFLOPS), this takes $\sim 0.13$ ms. Multiplied by $\lceil T / B \rceil \cdot K \approx 75 \cdot 5 = 375$ matmuls, the entire TP/FP
  computation is bounded below 50 ms on compute. In the original CPU-sparse path the same arithmetic took many seconds per chunk, repeated 40 times.

  The dense-on-GPU choice matters for two reasons:

  1. Memory access patterns. $B_k^{[b:b+B]}$ is $(4096, 10^4)$ float32 — 160 MB, laid out contiguously. cuBLAS achieves near-peak tensor-core throughput. Scipy
   SpGEMM, by contrast, is pointer-chasing through CSR indices with irregular memory access and no SIMD.
  2. Eliminating the transfer. Previously, each feature chunk required sae_feats.cpu().numpy() followed by sparse.csr_matrix(...) — a GPU→CPU copy of a $(T,
  250)$ tensor plus CSR construction, paid 40 times per shard. With loop inversion, $B_k$ lives and dies on the GPU.

  A byproduct: the SAE forward pass is now called $T/B \approx 75$ times instead of $40 \cdot 300 = 12{,}000$ times (the outer ×40 for feature chunks, the
  inner ×300 for token sub-batches of 1024). The total FLOPs for encoding are identical, but the fixed per-call overhead (normalization, allocation,
  torch.vstack) drops by two orders of magnitude. As you note, this isn't the dominant cost — but it's essentially free once you've inverted the loops, and it
  cleans up the critical path.

  Optimization 3: Domain Set Union via Boolean OR Accumulation

  $\mathrm{TP^{dom}}$ is harder to vectorize because "count distinct domain IDs caught" requires a set-union reduction, not a sum. The original implementation
  materialized the product $B_k \odot L_{:,c}$ and computed len(set(col.data)) per feature column — inherently Python, inherently slow.

  Key observation: $\mathrm{TP^{dom}}_{c,f,k}$ can be written as

  $$
  \mathrm{TP^{dom}}{c,f,k} = \sum{d \in \mathcal{D}c} \mathbf{1}\Bigl[; \exists, t; :; L{t,c} = d ,\wedge, A_{t,f} > \tau_k ;\Bigr]
  $$

  where $\mathcal{D}_c$ is the set of distinct domain IDs for concept $c$. Let $M^{(c)} \in {0,1}^{T \times |\mathcal{D}_c|}$ be the one-hot indicator for
  "token $t$ belongs to domain $d$". Then

  $$
  \bigl(M^{(c)\top} B_k\bigr){d, f} ;>; 0 \iff \exists, t, :, L{t,c} = d \wedge A_{t,f} > \tau_k
  $$

  The existential quantifier turns into a thresholded matmul. Across token batches, the "ever caught" relation accumulates via boolean OR:

  $$
  E_{c,k}^{(b+1)} ;=; E_{c,k}^{(b)} ;\lor; \bigl( M^{(c)\top}_{[b:b+B]} B_k^{[b:b+B]} > 0 \bigr)
  $$

  Finally $\mathrm{TP^{dom}}{c,f,k} = \sum_d E{c,k}[d, f]$.

  This trades a tight Python loop over $F$ for a GPU matmul of shape $(|\mathcal{D}_c|, B) \times (B, F)$ and a bitwise-OR into a persistent $(K,
  |\mathcal{D}_c|, F)$ boolean tensor. With $|\mathcal{D}_c| \sim 10^2$ and typical non-AA concept counts, the total auxiliary memory is a few hundred MB on
  GPU — easily within budget.

  Summary

  ┌─────────────────────────────┬───────────────────────────────────────────────────┬───────────────────────────────────────────────────────────┐
  │            Step             │                     Original                      │                         Optimized                         │
  ├─────────────────────────────┼───────────────────────────────────────────────────┼───────────────────────────────────────────────────────────┤
  │ TP/FP inner iterations      │ $40 \cdot K \cdot C = 10^5$ sparse CPU multiplies │ $\lceil T/B \rceil \cdot K \approx 375$ dense GPU matmuls │
  ├─────────────────────────────┼───────────────────────────────────────────────────┼───────────────────────────────────────────────────────────┤
  │ TP/FP backend               │ scipy.sparse element-wise + .sum(axis=0)          │ cuBLAS GEMM                                               │
  ├─────────────────────────────┼───────────────────────────────────────────────────┼───────────────────────────────────────────────────────────┤
  │ TP_dom backend              │ Python set() over column .data, $\sim 10^7$ calls │ Thresholded matmul + bitwise OR                           │
  ├─────────────────────────────┼───────────────────────────────────────────────────┼───────────────────────────────────────────────────────────┤
  │ SAE forward calls           │ $40 \cdot 300 = 12{,}000$                         │ $\lceil T/B \rceil \approx 75$                            │
  ├─────────────────────────────┼───────────────────────────────────────────────────┼───────────────────────────────────────────────────────────┤
  │ GPU↔CPU transfers per shard │ $\sim 40$ tensor copies + sparse conversions      │ 1 (final .cpu() on accumulators)                          │
  ├─────────────────────────────┼───────────────────────────────────────────────────┼───────────────────────────────────────────────────────────┤
  │ Empirical wall time         │ ~10 min/shard                                     │ dominated by the 75 SAE passes                            │
  └─────────────────────────────┴───────────────────────────────────────────────────┴───────────────────────────────────────────────────────────┘

  The largest gains come not from reducing FLOPs — several of these transformations leave asymptotic complexity unchanged — but from (a) replacing Python-level
   iteration with a single BLAS call, (b) moving a workload that is "sparse-ish but not sparse enough" from scipy.sparse on CPU to cuBLAS on GPU, and (c)
  restructuring the control flow so that GPU↔CPU transfers and sparse-matrix construction happen $\mathcal{O}(1)$ times per shard rather than
  $\mathcal{O}(F/\text{chunk})$.
