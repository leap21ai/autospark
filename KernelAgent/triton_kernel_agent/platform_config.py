# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Platform configuration registry for multi-backend support.

Usage:
    from triton_kernel_agent.platform_config import get_platform, get_platform_choices

    platform = get_platform("xpu")
    print(platform.device_string)  # "xpu"
    print(platform.guidance_block)  # Intel XPU-specific guidance
"""

from dataclasses import dataclass, field

DEFAULT_PLATFORM = "cuda"


@dataclass(frozen=True)
class PlatformConfig:
    """Configuration for a specific hardware platform/backend."""

    name: str
    device_string: str
    guidance_block: str
    kernel_guidance: str
    cuda_hacks_to_strip: tuple = field(default_factory=tuple)


# Platform-specific constants
_XPU_GUIDANCE = """\
**CRITICAL PLATFORM REQUIREMENTS FOR INTEL XPU:**
- Default tensor allocations to device='xpu' (never 'cuda'); CPU is allowed only when necessary.
- Check availability with: hasattr(torch, 'xpu') and torch.xpu.is_available()
- Do NOT monkey-patch torch.cuda or torch.device
- Do NOT set TRITON_BACKENDS environment variable
- Do NOT import or disable XPUDriver
- Use torch.xpu.synchronize() if synchronization is needed
- Intel XPU subgroup size is typically 16 (not 32 like CUDA warps)
- Preferred block sizes: 64, 128, 256, or 512"""

_XPU_KERNEL_GUIDANCE = """\
## Intel XPU-Specific Optimizations

You are generating a Triton kernel for Intel XPU (Xe GPUs). Follow these guidelines:

1. **Device Context**: Use 'xpu' as the device instead of 'cuda'
2. **Memory Hierarchy**: Intel Xe has different cache sizes - optimize accordingly
3. **Thread Configuration**:
   - Subgroup size is typically 8, 16, or 32 (flexible)
   - num_warps: typically 4, 8, or 16 for Intel GPUs
   - BLOCK_SIZE: prefer 64, 128, 256, or 512
4. **Optimal Block Sizes**: Start with 128-256 for most kernels
5. **Data Types**: Intel supports fp32, fp16, bf16 (fp8 varies by generation)"""

_GB10_KERNEL_GUIDANCE = """\
## NVIDIA GB10 (DGX Spark) — SM121 Hardware Constraints

You are generating a Triton kernel for the NVIDIA GB10 (DGX Spark, compute capability 12.1).
This GPU uses SM121 which is architecturally different from datacenter Blackwell (SM100).

**CRITICAL CONSTRAINTS — violating these will cause silent failures or crashes:**

1. **Shared memory: 128 KB per SM** (NOT 228 KB like datacenter Blackwell)
   - For matmul: BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N must fit in 128 KB
   - Safe autotune configs: max BLOCK_M=128, BLOCK_N=128, BLOCK_K=64
   - DO NOT use BLOCK_M=128 + BLOCK_N=256 combos (exceeds shared memory)

2. **No WGMMA** — SM121 uses Ampere-era `mma.sync` with registers, NOT datacenter Blackwell `tcgen05`
   - Warp specialization (`warp_specialize=True`) may not work — avoid unless tested
   - Persistent kernel patterns from H100 examples may fail

3. **No FlashAttention** — FA2/FA4 kernels are sm80-sm100 only
   - Use SDPA-style attention or manual online softmax instead
   - Do NOT use `from triton.tools.tensor_descriptor import TensorDescriptor` (TMA not available)

4. **No TMEM (Tensor Memory)** — 0 KB vs 256 KB on datacenter Blackwell

5. **Memory: 128 GB unified LPDDR5x** (CPU+GPU share same pool, 273 GB/s bandwidth)
   - Memory-bound kernels are common — optimize for bandwidth, not compute
   - No HBM — bandwidth is ~12x lower than H100

6. **Accumulation precision**: Always accumulate in FP32 for matmul/reductions
   - Use `tl.dot(a, b, acc, input_precision="ieee")` for BF16 matmul accuracy
   - Cast to output dtype only at the final store

7. **Autotune configs for SM121:**
   ```python
   @triton.autotune(
       configs=[
           triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=3, num_warps=4),
           triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=4, num_warps=4),
           triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=3, num_warps=4),
           triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=3, num_warps=4),
           triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 64}, num_stages=4, num_warps=2),
       ],
       key=[...],
   )
   ```

8. **48 SMs** — fewer than H100 (132) or A100 (108). Persistent kernels need smaller grid sizes.
"""

_GB10_CONSTRAINTS_TEXT = """\
   - GPU: NVIDIA GB10 (DGX Spark), compute capability 12.1 (SM121)
   - Shared memory per SM: 128 KB (NOT 228 KB like datacenter Blackwell)
   - NO WGMMA, NO TMEM, NO TMA tensor descriptors, NO FlashAttention
   - Uses Ampere-era mma.sync — treat like an Ampere GPU with newer numeric formats
   - Max safe autotune tile: BLOCK_M=128, BLOCK_N=128, BLOCK_K=64
   - Unified memory: 128 GB LPDDR5x at 273 GB/s (12x lower bandwidth than H100 HBM3)
   - 48 SMs, 192 tensor cores (5th gen), 6144 CUDA cores
   - Always accumulate in FP32; use input_precision="ieee" for tl.dot with BF16
   - Avoid warp_specialize=True and persistent kernel patterns from H100 examples
"""

_XPU_CUDA_HACKS = (
    "torch.cuda.is_available = lambda: True",
    "_orig_torch_device = torch.device",
    "_real_torch_device = torch.device",
    "def _fake_torch_device",
    "torch.device = _fake_torch_device",
    'os.environ["TRITON_BACKENDS"] = "cuda"',
    "from triton.backends.intel.driver import XPUDriver",
    "XPUDriver.is_available = classmethod(lambda cls: False)",
)

# Platform registry
PLATFORMS: dict[str, PlatformConfig] = {
    "cuda": PlatformConfig(
        name="cuda",
        device_string="cuda",
        guidance_block="",
        kernel_guidance="",
        cuda_hacks_to_strip=(),
    ),
    "cuda_gb10": PlatformConfig(
        name="cuda_gb10",
        device_string="cuda",
        guidance_block=_GB10_CONSTRAINTS_TEXT,
        kernel_guidance=_GB10_KERNEL_GUIDANCE,
        cuda_hacks_to_strip=(),
    ),
    "xpu": PlatformConfig(
        name="xpu",
        device_string="xpu",
        guidance_block=_XPU_GUIDANCE,
        kernel_guidance=_XPU_KERNEL_GUIDANCE,
        cuda_hacks_to_strip=_XPU_CUDA_HACKS,
    ),
}


def get_platform(name: str) -> PlatformConfig:
    """Get platform configuration by name."""
    if name not in PLATFORMS:
        available = ", ".join(sorted(PLATFORMS.keys()))
        raise ValueError(f"Unknown platform '{name}'. Available: {available}")
    return PLATFORMS[name]


def get_platform_choices() -> list[str]:
    """Get list of available platform names for CLI choices."""
    return sorted(PLATFORMS.keys())
