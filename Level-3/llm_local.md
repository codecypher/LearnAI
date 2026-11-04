# LLM Local & Self-Hosted

## 7 Approaches to Efficient LLM Training on Limited Hardware

Here are seven engineering techniques to train LLMs on local GPU [1]:

### Quantized Low-Rank Adaptation (QLoRA and DoRA)

**The Concept:** Freezing base model weights in an optimized 4-bit representation while injecting trainable low-rank, full-precision decomposition matrices into self-attention and feed-forward projection layers.

**How It Works:**

Base parameters are quantized into 4-bit NormalFloat (NF4), a distribution tailored to normally distributed neural network weights.

Double Quantization (DQ) quantizes the quantization constants themselves, saving an additional 0.37 bits per parameter.

During the forward pass, base weights are dynamically dequantized into BF16 for compute, added to the low-rank update matrix ΔW = B · A (scaled by α / r), and discarded from cache right away.

Weight-Decomposed Low-Rank Adaptation (DoRA) extends this approach by decoupling magnitude and directional updates to mirror full fine-tuning gradient trajectories.

**The Catch:** Dynamic on-the-fly dequantization introduces compute overhead that degrades training throughput (Tokens Per Second, or TPS) by 20% to 35% compared to native 16-bit training.

In addition, merging adapter weights back into base models for zero-latency serving requires dequantizing the base model back to 16-bit which prevents direct deployment in 4-bit environments without compound precision loss.

**When to Use It:** Fine-tuning 7B to 70B parameter models on single or dual consumer-grade 24 GB GPUs where aggregate VRAM can't fit unquantized model weights and gradient buffers.

### Memory-Aware Low-Rank Optimizers (GaLore)

**The Concept:** Full-parameter learning by projecting high-dimensional gradient matrices into a compact low-rank subspace, which cuts optimizer state memory footprint without freezing layers.

**How It Works:**

Standard AdamW maintains two FP32 states (first and second moments) per trainable parameter, consuming 8 bytes per parameter.

Gradient Low-Rank Projection (GaLore) applies Singular Value Decomposition (SVD) or randomized orthogonal projections to the gradient tensor G ∈ ℝm × n, tracking momentum and variance only for projected matrices P ∈ ℝm × r where r ≪ min(m, n).

Projections are updated periodically (every T steps) rather than per-iteration to amortize SVD computational overhead.

**The Catch:** Periodic SVD factorizations introduce compute stalls that cause step-latency spikes.

Hyperparameter selection is brittle: picking a bad subspace update frequency (T) or rank cutoff (r) destabilizes the optimization trajectory and can trigger sudden loss divergence mid-training.

**When to Use It:** Full-parameter pre-training or aggressive domain adaptation on memory-limited setups where parameter-efficient fine-tuning (LoRA) does not adapt well to complex out-of-domain feature distributions.

### Fully Sharded Data Parallelism With Host Memory Offloading (FSDP / ZeRO-3)

**The Concept:** Sharding optimizer states, gradients, and model parameters across both available device VRAM and system host RAM (CPU memory), paging tensors across PCIe buses strictly on demand.

**How It Works:**

Under ZeRO-Stage 3 / FSDP Full Shard, each GPU holds only 1/N of the complete model state during idle intervals.

During the forward pass, an All-Gather collective communication reconstructs layer weights right before computation and deallocates them once execution advances to the next layer.

In host-offload mode, non-active parameter shards and optimizer states reside in pinned host CPU RAM, streaming over the PCIe bus asynchronously via non-blocking CUDA streams concurrently with compute kernels.

**The Catch:** Offloading using consumer PCIe Gen4/Gen5 lanes creates severe I/O bottlenecks.

When GPU compute finishes before host-to-device (H2D) tensor transfers complete, the SMs (Streaming Multiprocessors) idle in wait states, dropping GPU compute utilization below 30%.

In addition, PCIe bandwidth contention often starves dataloader worker processes streaming fresh training batches from NVMe drives.

**When to Use It:** Scaling training runs for models whose parameter count exceeds the total aggregate VRAM of your multi-GPU node (such as training a 30B+ parameter model using four 24 GB GPUs).

### Selective Activation Checkpointing and Recomputation

**The Concept:** Dropping high-memory intermediate activation tensors from VRAM during the forward pass and selectively recomputing them during the backward autograd pass.

**How It Works:**

Standard backpropagation stores every intermediate activation tensor generated during the forward pass to evaluate the chain rule gradients.

Selective activation checkpointing identifies memory-heavy, compute-cheap operations (such as GeLU/SwiGLU activations, layer norms, and dropout masks) and discards them after the forward computation.

During the backward pass, these tensors are re-evaluated on the fly from the nearest retained checkpointed boundary (typically the transformer block boundary).

**The Catch:** Full activation recomputation adds roughly 30% computational overhead to total FLOPs per training step.

If implemented naively without profiling tensor allocation life-cycles, frequent memory deallocations and reallocations trigger severe CUDA memory fragmentation, causing sudden CUDA out of memory errors even when reported gross VRAM usage sits below hardware limits.

**When to Use It:** Training with long context windows (8k to 32k+ tokens) where activation memory footprint scales linearly or quadratically and eclipses static weight allocations.

### Hardware-Aware Memory-Tiled Kernels (FlashAttention-2 and Fused Operations)

**The Concept:** Restructuring attention computation and elementwise operations to execute entirely within high-bandwidth on-chip SRAM, bypassing redundant reads and writes to high-latency GPU HBM (High Bandwidth Memory).

**How It Works:**

Standard attention materializes the full N × N attention matrix S = QKT in HBM, generating massive read/write traffic.

FlashAttention-2 tiles the Query, Key, and Value matrices into blocks that fit within the GPU's L1 cache/SRAM, computing softmax normalization incrementally via online scaling without ever writing the full attention matrix to global memory.

Fused kernels combine LayerNorm, bias additions, and activation functions into single CUDA kernel launches, minimizing memory transfer round-trips.

**The Catch:** Custom fused kernels are tightly coupled to specific GPU microarchitectures (such as Ada Lovelace, Hopper, Ampere) and specific compute capability flags.

Compiling FlashAttention on non-standard consumer drivers or custom containerized environments often triggers ABI incompatibility issues, silent fallback to slow un-fused PyTorch native kernels, or precision underflow on unaligned sequence lengths without proper padding masks.

**When to Use It:** This is a must for all transformer training workloads regardless of hardware scale, to maximize SM occupancy and eliminate memory bandwidth bottlenecks.

### Mixed-Precision Training With FP8 (E4M3/E5M2) Formats

**The Concept:** Running tensor contractions and matrix multiplications using 8-bit floating-point representations, cutting memory bandwidth consumption and activation buffer sizes by half compared to 16-bit formats.

**How It Works:**

Mixed-Precision Training With FP8 uses two distinct FP8 representations: E4M3 (1 sign bit, 4 exponent bits, 3 mantissa bits) for activations and weights to prioritize numerical precision, and E5M2 (1 sign bit, 5 exponent bits, 2 mantissa bits) for gradients to accommodate wider dynamic range.

Dynamic scaling factors are computed per-tensor or per-tile at runtime to prevent underflow and overflow before casting values into FP8 Tensor Cores.

**The Catch:** FP8's dynamic range is narrow. Without rigorous delayed-scaling algorithms or per-channel quantization schemes, gradient vanishing occurs during backward passes on deeper layers, leading to unrecoverable training divergence and loss explosion.

FP8 hardware acceleration is also limited to modern microarchitectures (Ada Lovelace / Hopper and newer).

**When to Use It:** Training on modern Ada Lovelace (RTX 4090, L40S) or Hopper (H100) hardware where FP8 Tensor Cores can double compute throughput and halve activation VRAM.

### Sequence Chunking and RingAttention Over Commodity Interconnects

**The Concept:** Distributing ultra-long context sequences across multiple devices by passing Query, Key, and Value blocks in a ring topology concurrently with attention computation.

**How It Works:**

Rather than fitting an entire 64k+ context sequence on a single GPU's memory buffer, RingAttention splits the sequence along the temporal dimension across K devices.

Device _i_ computes attention between its local Query block and local Key/Value block, then kicks off an asynchronous non-blocking P2P ring communication to send its KV block to device (i+1) mod K while receiving from (i-1) mod K.

Compute and network communication overlap entirely, eliminating the need for high-end NVLink meshes.

**The Catch:** On consumer hardware running over standard PCIe buses or 1GbE/10GbE local network interfaces, communication latency significantly outpaces compute time for small batch sizes.

If network transfer time exceeds the block compute time, the pipeline stalls at every ring step, wiping out throughput gains.

**When to Use It:** Scaling training context windows beyond 32k tokens on distributed multi-node or multi-GPU setups that lack dedicated high-bandwidth NVLink bridges.

### Summary

Long-running training operations on constrained hardware will eventually surface silent failure modes that benchmarks miss: non-deterministic CUDA kernel behavior across driver versions, thermal throttling on consumer-grade hardware under sustained 100% duty cycles, and checkpoint corruption from asynchronous disk I/O bottlenecks.

Production pipelines need continuous metric tracing of floating-point underflow rates, GPU PCIe bus utilization counters, and automated gradient checkpoint verification hooks to prevent hundreds of compute hours from being wasted on silently diverged weights.

LLM training on limited hardware comes down to memory hierarchy management rather than brute-force compute scaling.

By decoupling weight precision, optimizer state tracking, and activation persistence through QLoRA, GaLore, and FlashAttention-2, it is possbile to reach convergence parity with enterprise-scale compute clusters on a fraction of the hardware cost.

## References

[1]: https://www.kdnuggets.com/7-approaches-to-efficient-llm-training-on-limited-hardware "7 Approaches to Efficient LLM Training on Limited Hardware"

[Tweaking Local Language Model Settings with Ollama](https://www.kdnuggets.com/tweaking-local-language-model-settings-with-ollama)

[vLLM](https://docs.vllm.ai/en/latest/)

[Fine tune your own custom LLM with Canonical Charmed Kubeflow and Feast](https://ubuntu.com/blog/fine-tune-your-own-custom-llm-with-canonical-charmed-kubeflow-and-feast?via=dailydev)
