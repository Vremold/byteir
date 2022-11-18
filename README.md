# ByteIR

ByteIR is an MLIR-based compiler for CPU/GPU/ASIC.
It is an umbrella repo for all related IRs and passes.


## IRs (Dialects)

### ACE
ACE is a supplement to MHLO dialect for composite operators.

### ByRE (ByteDance's Representation for Execution)
ByRE is a runtime IR for the ByteIR runtime. 

### MHLO 
An external dialect from https://github.com/tensorflow/mlir-hlo.

## Dependency 
***LLVM/MLIR***: https://code.byted.org/byteir/llvm-build, current llvm commit id: 74fb770de9399d7258a8eda974c93610cfde698e

***mhlo_tools***: https://tosv.byted.org/obj/turing/byteir/mhlo_tools-1.0.2-cp37-cp37m-linux_x86_64.whl

***Python*** (for python binding): minimum version is 3.6, requiring numpy and pybind11 installed.

## Build
### Linux/Mac 
```bash
python3 -m pip install -r requirements.txt

mkdir /path_to_byteir/build
cd /path_to_byteir/build/

# build ByteIR
cmake ../cmake/ -G Ninja \
                -DCMAKE_BUILD_TYPE=Release \
                -DLLVM_INSTALL_PATH=path_to_LLVM_installed_or_built_directory \
                -DLLVM_EXTERNAL_LIT=lit_executatble_location # or using $(which lit), this is optional for external lit 

cmake --build . --config Release
```
### Windows 
```bash
mkdir /path_to_byteir/build
cd /path_to_byteir/build/

# build ByteIR
cmake ../cmake/ -G "Visual Studio 16 2019" -A x64 \
                -DCMAKE_BUILD_TYPE=Release \
                -DLLVM_INSTALL_PATH=path_to_LLVM_installed_or_built_directory \
                -DLLVM_EXTERNAL_LIT=lit_location # this is optional for external lit 

cmake --build . --config Release
```

## Testing 
This command runs all ByteIR unit tests:
```
cmake --build . --config Release --target check-byteir
```
ByteIR relies on ```llvm-lit``` and ```FileCheck``` for testing.
For more information, you can refer to [this page](https://www.llvm.org/docs/CommandGuide/FileCheck.html)
All the tests are placed in the folder ```byteir/test```.

## Install (Optional)
```bash
cmake --install . --prefix path_to_install_BYTEIR
```

## Passes
Useful Pass Description [doc/passes.md](doc/passes.md)

### Examples
Mhlo-to-NVVM pass pipeline [doc/mhlo-to-nvvm.md](doc/mhlo-to-nvvm.md)

## License (TODO)
