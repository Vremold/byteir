# Typical Pass Pipeline for Mhlo to NVVM

Here we show an example going through mhlo->linalg->affine->gpu->nvvm

## High IR (Mhlo)
### High-IR Passes
High-IR passes like fusion happens first before lowering high-IR (mhlo).
Here is a typical pass pipeline for fusing elementwise op. 
A tag is attached for limiting applicable region or function in later passes.

```bash
 -fuse-element="attach-tag=__byteir_elementwise_fusion__" 
```

## First Mid IR (Linalg)
### Preparing lowering mlo to linalg
Several high-IR type might need to be rewritten before lowering a mid-IR. The Tuple, in our case, mhlo, need to be flattened (in a body) and expanded (in arguments). 
The entry-function's name is specified to expanding return tuple. 

```bash
-expand-hlo-tuples="entry-function=main" -cse -mhlo-flatten-tuple -cse
```

Also depending on lowering passes, some ops might be rewritten too. The mhlo.FusionOp, in our case, needs to be outlined into another private function to providing its own function for later limiting applicable region. 
 
```bash
-fusion-outlining -cse 
```

### Lowering to a mid-IR (Linalg)

In this case, we lower to Linalg for elementwise fusion patterns. Here we only apply lowering in the limited applicable function by using the constraint parameter.

Depending on lowering passes, some necessary passes might be required with lowering. 
In our case, an unrealized cast need to be resolved into a scalar form into Linalg. 

```bash
-hlo-fusion-to-linalg="anchor-tag=__byteir_elementwise_fusion__" -unrealized-cast-to-linalg -cse
```

### Mid-IR Passes (Linalg)
Mid-IR passes like fusion or tiling happens here. 
In our case, linalg elementwise fusion can give loop fusion results if meeting tensor shape requirement. 

```bash
-linalg-fuse-elementwise-ops -cse
```

## Second Mid IR (Affine)
### Preparing lowering Linalg to Affine
Same as high-IR, some preparing passes are required, and for a specific source-and-target-IR pair.
In our case, LinAlg-to-Affine, bufferiation and detensorization are needed.

```bash
-linalg-bufferize -func-bufferize -cse -sccp -linalg-detensorize -cse
```

### Lowering to another mid-IR (Affine)

```bash
-convert-linalg-to-affine-loops
```

### Mid-IR Passes (Affine)
Here, loop coalescing is performed for 1) providing more loop fusion possibility across different tensor shapes and 2) better and simpler loop structure for our backend, considering this is a elementwise fusion case. 

Affine structure simplification is for better affine map recognition in later passes. It is typically called right after a affine loop transformation.

Note after loop fusion, we apply loop coalescing and affine structure simplification again in case loop fusion generating nested loops.

Also, CMAE is performed to eliminate redundant load and store (within a basic block) generated from loop fusion. 

```bash
-loop-coalescing -simplify-affine-structures -affine-loop-fusion -loop-coalescing -simplify-affine-structures -cmae -cse -cse
```

## First Low IR (GPU/Affine)
### Preparing lowering Affine loop to GPU

Affine dialect contains both loop ops and access ops.
We typically lowering outer loops first, and then lowering inner loop and access ops later unless there is a conflict in the lowering pass implmenetation. (Note this is implementation dependent, not input dependent.)

In our case (using default mlir's lowering Affine loop to GPU), 
affine load and store rewriting is needed before loop lowering. 

Since our example is elementwise, which applied loop coalescing, so there will be only a single loop (meaning outer loop).

```bash
-rewrite-affine-to-memref
```

### Lowering to a low IR (GPU/Affine)
This stage lowers a coalesced for loop to GPU. 

```bash
-coalesced-for-to-gpu -cse -sccp -cse 
```

### Low-IR Passes (GPU)
In this starge, we only outline gpu kernel.

```bash
-gpu-kernel-outlining -cse
```

## Output Low IR (NVVM)
### Preparing lowering GPU/Affine to NVVM 
In this starge, we prepre lowering to NVVM by lowering the rest general ops in bodies through SCF, then STD.

```bash
-lower-affine -cse -convert-scf-to-std
```

### Lowering to output low-IR (NVVM)
In this starge, we lower to NVVM. Scalar unrealize cast will need to be removed in a valid NVVM.

```bash
-gpu-to-nvvm-ext -cse -reconcile-unrealized-casts -cse 
```
