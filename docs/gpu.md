# GPU Acceleration

## Performance

### NOTE: as of 5/11 these numbers are out of date

Measured throughput on real hardware:

| | Throughput | Hardware |
|-|------------|----------|
| **CPU** | ~1.3M keys/sec | AMD Ryzen 9 7950X3D (32 threads) |
| **GPU** | ~68M keys/sec | NVIDIA RTX 4090 (24GB) |

The GPU is roughly **50x faster** than the CPU on this hardware. The speedup comes from running the full Ed25519 pipeline across thousands of GPU threads in parallel.

## Building with CUDA

```bash
cargo build --release --features cuda
```

Requires the NVIDIA CUDA Toolkit to be installed. The CUDA kernel is compiled at runtime via NVRTC, so `nvrtc` and `cuda` shared libraries must be on PATH.

The `cudarc` dependency is configured for CUDA 13.1 by default. To target a different CUDA version, change the `cuda-13010` feature in `Cargo.toml` to match your installation (e.g. `cuda-12060` for CUDA 12.6).

## How GPU mode works

When `--gpu-only` (or hybrid) is selected, the entire keygen pipeline runs on the GPU. A single host thread loops: launch kernel batch, sync, check for match, repeat. The TUI progress display works identically for both CPU and GPU modes.

Each `GpuSearcher` draws a fresh 32-byte `start_scalar` from the OS CSPRNG and advances it by `8 * keys_checked` between batches. Per launch, every thread runs **one** `ge_scalarmult_base` on `start_scalar + 8 * tid` to derive its starting point `A`, then iterates `ITERS_PER_THREAD` (256 on CUDA, 1024 on Metal) cheap point additions of `8B` (using `base[0][7] = 8B` as a `ge_precomp`) — each iteration produces a fresh candidate pubkey from `A` and checks the prefix. To amortize the expensive `fe_invert` needed for point compression, the kernel uses **Montgomery batch inversion** with `VANITY_BATCH` (16 on CUDA, 256 on Metal): one inversion per batch instead of one per point. On Metal the inversion dominated at B=16; B=256 with 1024 iters nearly doubled throughput (M2 Pro: 100 → 192 MH/s). The Metal kernel then walks the chain on the Montgomery curve with x-only differential addition (`A_{i+1} = xADD(A_i, 8B, A_{i-1})`, 4M+2S per step instead of the 7M Edwards mixed add) and maps back with `y = (u-1)/(u+1)` inside the same batch inversion; the x sign bit is unavailable, so the host rebuilds the full pubkey from the scalar and checks the low 255 bits (M2 Pro: 192 → 219 MH/s). On every prefix hit the kernel writes back the scalar and pubkey, and the host runs a defensive `scalar·B == pubkey` check via curve25519-dalek before accepting the match.

The Ed25519 field arithmetic uses the ref10 implementation (radix-2^25.5, int32[10] limbs) from [solana-perf-libs](https://github.com/solana-labs/solana-perf-libs), which is a CUDA adaptation of the [SUPERCOP](https://bench.cr.yp.to/supercop.html) reference code. Scalar multiplication uses windowed lookup tables with 32 precomputed basepoint multiples.

## Verification

Use `--verify` to cross-check GPU-generated keys against the CPU implementation. This is useful for validating correctness after modifying the CUDA kernel.

Use `--benchmark <SECS>` to run a separate `vanity_count_matches` kernel that never exits early on a hit. It atomically tallies every match over the given duration and compares the observed count against the expected count implied by the reported keys/sec rate — a sanity check that the reported throughput isn't inflated by mid-launch early exits. Works on both the CUDA and Metal backends.

## Sources

- [solana-perf-libs](https://github.com/solana-labs/solana-perf-libs) — CUDA Ed25519 field arithmetic (ref10)
- [cudarc](https://github.com/coreylowman/cudarc) — safe Rust bindings for the CUDA driver API
