set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name ByteIRPipelines)
add_public_tablegen_target(ByteIRPipelinesPassIncGen)
# add_mlir_doc(Passes PipelinesPasses ./ -gen-pass-doc)