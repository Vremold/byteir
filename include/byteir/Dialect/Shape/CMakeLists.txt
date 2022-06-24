add_mlir_dialect(ShapeExtOps shape_ext)
add_mlir_doc(ShapeExtOps ShapeExtOps Dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name ByteIRShape)
add_public_tablegen_target(ByteIRShapePassIncGen)
