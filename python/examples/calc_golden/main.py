import argparse
import logging

from byteir.ir import register_all_backends
from byteir.utils.golden_ref import MhloGoldenRefGenerator

try:
    # register all builtin backends(i.e. mhlo dialect w/o custom call and TF dialect)
    register_all_backends()
except ImportError:
    logging.error("Failed to register all backends\n")
    raise

def do_main():
    logging.basicConfig(level=logging.WARN)

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=str, help='path to input mlir module')
    args = parser.parse_args()

    # create golden reference generator on given model path
    gen = MhloGoldenRefGenerator.load_from_file(args.input)
    # generate random input and calculate golden reference for every operation outputs 
    gen.generate()
    # mapping from mlir value to numpy ndarray is stored in value2tensor
    print(gen.value2tensor)

if __name__ == "__main__":
    do_main()
