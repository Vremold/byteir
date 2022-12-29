# ByteIR Linalg Extension

ByteIR compiler extends the MLIR linalg dialect to support several non-trivial patterns.
ByteIR implements in a way of introducing a linalg-ext dialect on top of the existing linalg dialect.
Ops and transformations in linalg-ext are expected to work interchangeably with existing ones in linalg, and expected to eventually be upstreamed to LLVM.

## Rationales
### Need of non-trivial patterns for linalg

Several performance-critical patterns are still not covered well in the upstream linalg.
Some of those patterns might not be easily expressible in the linalg dialect either through generic ops or even only relying on existing linalg interfaces. Top-K and Scan (cumsum) might belong to this category. 

Some might be expressible, through composing several generic ops, but might obstruct desired transformations due to lack of proper interfaces. Softmax belongs to this category.


### Implementation of introducing linalg-ext

Introducing linalg-ext can provide several benefits as follows,
* it clearly separate the extension of ops or transformations from the existing linalg, avoiding misusing.
* it can intuitively resolve the patterns that require introducing interfaces.


## Transformation Extension

Several transformations are enhanced or introduced in ByteIR linalg-ext. 

***Tile label transformation*** is introduced 
* to indicate loop type (parallel or reduction) through attributes.

Note this tile label transformation also work with existing linalg tile and fuse transformation.

***Tile transformation*** is enhanced 
* to support linalg-ext ops.

***Fuse transformation*** is enhanced
* to support linalg-ext ops,
* to support intermediates as outputs within a fusion.

Here shows the difference when there is an intermediate as as output.
```
// input.mlir
func.func @fuse_element(%arg0: tensor<512x128xf32>, %arg1: tensor<512x128xf32>) -> (tensor<512x128xf32>, tensor<512x128xf32>) {
  %0 = linalg.elemwise_unary ins(%arg0 : tensor<512x128xf32>)
                             outs(%arg1: tensor<512x128xf32>) -> tensor<512x128xf32>
  %1 = linalg.elemwise_binary ins(%0, %arg0 : tensor<512x128xf32>, tensor<512x128xf32>)
                             outs(%arg1: tensor<512x128xf32>) -> tensor<512x128xf32>
  return %0, %1 : tensor<512x128xf32>, tensor<512x128xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.elemwise_binary"]} in %arg1
  %1, %loops:2 = transform.structured.fuse %0 {tile_sizes = [32, 32], tile_interchange = [0, 1]}
}

// result after transform.structured.fuse 
func.func @fuse_element_static(%arg0: tensor<512x128xf32>, %arg1: tensor<512x128xf32>) -> (tensor<512x128xf32>, tensor<512x128xf32>) {
  %c128 = arith.constant 128 : index
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %0 = linalg.elemwise_unary ...  // duplicate elemwise_unary
  %1 = scf.for %arg2 = %c0 to %c512 step %c32 iter_args(%arg3 = %arg1) -> (tensor<512x128xf32>) {
    %2 = scf.for %arg4 = %c0 to %c128 step %c32 iter_args(%arg5 = %arg3) -> (tensor<512x128xf32>) {
      ...
      %3 = linalg.elemwise_unary ...
      ...
      %4 = linalg.elemwise_binary ...
      %inserted_slice = tensor.insert_slice ...
      scf.yield %inserted_slice : tensor<512x128xf32>
    }
    scf.yield %2 : tensor<512x128xf32>
  }
  return %0, %1 : tensor<512x128xf32>, tensor<512x128xf32>
}

// result after transform.structured.fuse_ext
func.func @fuse_element_static(%arg0: tensor<512x128xf32>, %arg1: tensor<512x128xf32>) -> (tensor<512x128xf32>, tensor<512x128xf32>) {
  %c128 = arith.constant 128 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  %0:2 = scf.for %arg2 = %c0 to %c512 step %c32 iter_args(%arg3 = %arg1, %arg4 = %arg1) -> (tensor<512x128xf32>, tensor<512x128xf32>) {
    %1:2 = scf.for %arg5 = %c0 to %c128 step %c32 iter_args(%arg6 = %arg1, %arg7 = %arg1) -> (tensor<512x128xf32>, tensor<512x128xf32>) {
      ...
      %2 = linalg.elemwise_unary ins(%extracted_slice : tensor<32x32xf32>) outs(%extracted_slice_0 : tensor<32x32xf32>) -> tensor<32x32xf32>
      ...
      %3 = linalg.elemwise_binary ins(%2, %extracted_slice_1 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%extracted_slice_2 : tensor<32x32xf32>) -> tensor<32x32xf32>
      %inserted_slice = tensor.insert_slice ...
      %inserted_slice_3 = tensor.insert_slice ...
      scf.yield %inserted_slice, %inserted_slice_3 : tensor<512x128xf32>, tensor<512x128xf32>
    }
    scf.yield %1#0, %1#1 : tensor<512x128xf32>, tensor<512x128xf32>
  }
  return %0#1, %0#0 : tensor<512x128xf32>, tensor<512x128xf32>
}
```

***Elementwise fusion transformation*** is enhanced
* to support intermediates as outputs within a fusion,
* to support both producer-consumer fusion and input-sharing fusion,
* to support map fusion by automatically converting map ops to generic ops.


Here shows the difference when there is an input-sharing fusion
```
// input.mlir
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @input_sharing(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %3 = arith.addf %in, %in_1 : f32
    linalg.yield %3 : f32
  } -> tensor<?x?xf32>
  %2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %3 = arith.mulf %in, %in_1 : f32
    linalg.yield %3 : f32
  } -> tensor<?x?xf32>
  return %1, %2 : tensor<?x?xf32>, tensor<?x?xf32>
}

// result after linalg-fuse-elementwise-ops, unchanged
func.func @input_sharing(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %3 = arith.addf %in, %in_1 : f32
    linalg.yield %3 : f32
  } -> tensor<?x?xf32>
  %2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %3 = arith.mulf %in, %in_1 : f32
    linalg.yield %3 : f32
  } -> tensor<?x?xf32>
  return %1, %2 : tensor<?x?xf32>, tensor<?x?xf32>
}

// result after linalg-fuse-elementwise-ext="shared-input"
func.func @input_sharing(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
  %1:2 = linalg.generic {indexing_maps = [#map, #map, #map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%0, %0 : tensor<?x?xf32>, tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_1: f32, %in_2: f32, %out: f32, %out_3: f32):
    %2 = arith.addf %in, %in_1 : f32
    %3 = arith.mulf %in, %in_2 : f32
    linalg.yield %2, %3 : f32, f32
  } -> (tensor<?x?xf32>, tensor<?x?xf32>)
  return %1#0, %1#1 : tensor<?x?xf32>, tensor<?x?xf32>
}
```

## Linalg-ext Op Extension

### Alias Op

### Diag Op

### Softmax Op

#### Flash attention using Softmax




