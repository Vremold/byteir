## golden reference calculator
### Usage
  python3 exmples/calc_golden/main.py --input MODEL_PATH

  It was just a demo on how to use `MhloGoldenRefGenerator`

### Supported dialects and requirements

  all of following requirements should be located in PYTHONPATH

  | Dialect | Requirements |
  | ---- | ---- |
  | mhlo w/o custom_call | MHLO python binding, prebuilt tf_mlir_exit.so |
  | TF | prebuilt tf_mlir_exit.so |
  | mhlo.custom_call("ftv4.*") [WIP] | pytorch, prebuilt th_fastertransformer.so |

### Some of details

  - use numpy ndarray as the key data structure of input/output tensors of IRExecutor and golden references:

    - most of ML/DL frameworks could exchange data with numpy
    
    - it was quite easy to write custom evaluation rules based on numpy array (see dispatcher rules in ir_executor.py)

  - pass mlir operation to tf/mhlo dialect backend in textual format instead of in-memory object, since keep ABI-level compatibility between byteir and tf-frontend is too hard
