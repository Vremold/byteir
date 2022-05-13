set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name ByteIRGPUPipelines)
add_public_tablegen_target(ByteIRGPUPipelinesPassIncGen)
# add_mlir_doc(Passes PipelinesPasses ./ -gen-pass-doc)