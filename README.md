# The ByteIR Project

The ByteIR Project is a BytyeDance model compilation solution.
ByteIR includes compiler, runtime, and frontends, and provides an end-to-end model compilation solution.

Although all ByteIR components (compiler/runtime/frontends) are together to provided an end-to-end solution, and 
all under the same umbrella of this repository, 
each component technically can perform independently.

### Project Status
ByteIR is still in its early phase. 
In this phase, we are aiming to provide well-defined, necessary building blocks and infrastructure support for model compilation in a wide-ranage of deep learning accelerators as well as general-purpose CPUs and GPUs.
Therefore, highly-tuned kernels for specific achiecture might not have been prioritized. 
For sure, any feedback for prioritizing specific achiecture or correpsonding contribution are wellcome.

## [Compiler](compiler/README.md)

ByteIR Compiler is an MLIR-based compiler for CPU/GPU/ASIC.

## [Runtime](runtime/README.md)

ByteIR Runtime is a common, lightweight runtime, capable to serving both existing kernels and ByteIR compiler generated kernels.

## [Frontends](frontends/README.md)

ByteIR Frontends includes Tensorflow, PyTorch, and ONNX.

## [LICENSE](LICENSE)