import pytest
from test.base import TestBase


class TestModelsNumeric(TestBase):

    @pytest.fixture(autouse=True)
    def setup(self, tmpdir_factory):
        self.data_dir = "../../../bdaimodels/onnx/onnx_frontend/"  # ../.. is byteir's project root
        tmp_dir = tmpdir_factory.mktemp(self.data_dir.replace("/", "_"))
        self.tmp_dir = str(tmp_dir)

    def test_mnist(self):
        input_shape_dtype = [
            ["image", (1, 1, 28, 28), "float32"],
        ]
        self.run(model_filename="mnist_sim.onnx", input_shape_dtype=input_shape_dtype)

    def test_resnet(self):
        input_shape_dtype = [
            ["data", (8, 3, 224, 224), "float32"],
        ]
        self.run(model_filename="resnet50-v1-7_sim.onnx", input_shape_dtype=input_shape_dtype)

    def test_transformer(self):
        input_shape_dtype = [
            ["fbank", (16, 128, 80), "float32"],
        ]
        self.run(model_filename="transformer_encoder_sim.onnx", input_shape_dtype=input_shape_dtype)
