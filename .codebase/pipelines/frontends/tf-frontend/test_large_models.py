import os
import tempfile
import subprocess
from pathlib import Path
import numpy as np
import tensorflow as tf

import pdb

from mhlo_tools.ir_executor import Interpreter
from mhlo_tools.ir_executor.helper import mlir_attr_to_pyobj, mlir_type_to_dtype
from mhlo_tools.mlir import ir

import random
random.seed(12345)

TF_CUSTOMCALL_OPS = [
    "softmax",
    "log_softmax",
    "gelu",
    "erf",
    "arg_max",
    "arg_min",
    "top_k",
    "layer_norm",
    "l2_norm",
    "one_hot",
    "addn",
    "DynamicPartition",
    "DynamicMaskStitch",
    "DynamicStitch",
]


LARGE_MODEL_PATH = os.environ["TF_LARGE_MODEL_PATH"]
TF_FRONTEND = os.environ["TF_FRONTEND_BIN_PATH"]
MODEL_LIST = [
    {'model_name': "resnet/resnet50_v1.pb", 'batch_size': 32, 'output_names': ["softmax_tensor"], 'custom_compilation_cfgs': []},
    {'model_name': "bert/bert-base-mrpc-without-preprocess.pb", 'batch_size': 32, 'output_names': ["loss/Softmax"], 'custom_compilation_cfgs': []},
    {'model_name': "recommender_model/industry_ecom_cvr_project_v50_spu_fm_bias_batch_r2042581_0.pb", 'batch_size': 100,
     'output_names': ["oracle_pred"], 'custom_compilation_cfgs': ["-staticalize-dynamic-shape=True",
                                                                  "-remove-control-flow=True",
                                                                  "-force-set-batch-size=False",
                                                                  "-customcall-ops={}".format(",".join(TF_CUSTOMCALL_OPS))]}
]
# MODEL_LIST=[("quicksilver/nlp/bernard_model_27285.pb", 100, [])]

def read_pb_model(pb_modle_path):
    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(pb_modle_path, "rb") as f:
        graph_def.ParseFromString(f.read())
    return graph_def

def calculate_tf_outputs(pb_model_path, inputs_name, outputs_name, inputs):
    graph_def = read_pb_model(pb_model_path)
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    with tf.compat.v1.Session(graph=graph) as sess:
        feed_dict = {graph.get_tensor_by_name(name + ":0"): value for name, value in zip(inputs_name, inputs)}
        output_tensors = [graph.get_tensor_by_name(name + ":0") for name in outputs_name]
        outputs = sess.run(output_tensors, feed_dict=feed_dict)
    return outputs

def translate_from_tf_graph(model_path, batch_size, output_names, custom_compilation_cfgs):
    cmd_opts = [TF_FRONTEND]
    cmd_opts += [model_path]
    cmd_opts += ["-batch-size={}".format(batch_size)]
    cmd_opts += ["-tf-output-arrays={}".format(",".join(output_names))]
    cmd_opts += ["-mlir-print-debuginfo"]
    cmd_opts.extend(custom_compilation_cfgs)
    cmd_opts += ["-o"]
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "temp.mhlo.mlir"
        cmd_opts += [path]
        subprocess.check_call(cmd_opts)
        with open(path, "r") as f:
            return f.read()

def generate_inputs(entry_func):
    inputs = []
    for arg in entry_func.arguments:
        shaped_type = ir.ShapedType(arg.type)
        shape = shaped_type.shape
        dtype = mlir_type_to_dtype(shaped_type.element_type)
        if dtype is np.str_:
            inputs.append(np.ndarray(shape=shape, dtype=np.str_))
        elif dtype in [np.int32, np.uint32, np.int64, np.uint64]:
            inputs.append(np.random.randint(0, 2, size=shape).astype(dtype))
        else:
            inputs.append(np.random.normal(loc=0.0, scale=0.1, size=shape).astype(dtype))
    return inputs

def calculate_mhlo_golden(mhlo_str):
    ENTRY_FUNC_KEY = "byteir.entry_point"
    with Interpreter.load_from_string(mhlo_str, True) as interp:
        module = interp._mod
        entry_func = ir.SymbolTable(module.operation)["main"]
        entry_point_dict = mlir_attr_to_pyobj(entry_func.attributes[ENTRY_FUNC_KEY])
        inputs_name = entry_point_dict["inputs"]
        inputs = generate_inputs(entry_func)
        outputs = interp.call_function("main", inputs)
    return inputs_name, inputs, outputs

def test_large_models():
    for model in MODEL_LIST:
        print(model["model_name"])
        mhlo_str = translate_from_tf_graph(os.path.join(LARGE_MODEL_PATH, model["model_name"]), model["batch_size"], model["output_names"], model["custom_compilation_cfgs"])
        inputs_name, inputs, mhlo_outputs = calculate_mhlo_golden(mhlo_str)
        tensorflow_outputs = calculate_tf_outputs(os.path.join(LARGE_MODEL_PATH, model["model_name"]), inputs_name, model["output_names"], inputs)
        assert len(mhlo_outputs) == len(tensorflow_outputs)
        np.testing.assert_almost_equal(mhlo_outputs, tensorflow_outputs, decimal=2)

if __name__ == "__main__":
    test_large_models()
