set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name ByteIRHostPipelines)
add_public_tablegen_target(ByteIRHostPipelinesPassIncGen)
# add_mlir_doc(Passes PipelinesPasses ./ -gen-pass-doc)