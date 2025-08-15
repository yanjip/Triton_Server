import json
import os
import numpy as np
from towhee.serve.triton.bls.python_backend_wrapper import pb_utils
from tritonclient.grpc import model_config_pb2
from google.protobuf import text_format
import logging
logging.basicConfig(filename='/workspace/log_res/onnx.log', level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
class TritonPythonModel:
    """
    一个简单的文本切分模型。
    输入一个长字符串，按固定长度切分成一个字符串列表。
    """
    def initialize(self, args):
        """
        模型初始化时调用。
        """
        print("Text Splitter model initialized")


    def execute(self, requests):
        responses = []
        # 遍历所有请求
        for request in requests:
            # 1. 获取输入张量
            in_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT")

            # 2. 将张量数据解码为UTF-8字符串
            # 输入shape为[1]，所以我们取第一个元素
            long_text = in_tensor.as_numpy()[0].decode('utf-8')

            # 3. 执行切分逻辑（这里使用简单的固定长度切分作为示例）
            chunk_size = 15  # 定义每个切片的最大长度
            chunks = [long_text[i:i + chunk_size] for i in range(0, len(long_text), chunk_size)]
            logging.warning(f"切分后的原始列表: {chunks[:4]} ...")  # 确认是['str1', 'str2', ...]

            # 核心：创建一维object数组，每个元素是独立字符串
            chunks_np = np.array(chunks, dtype=object)  # 直接转换列表为数组，确保每个元素独立
            # 验证形状是否为一维 [N]
            logging.warning(f"chunks_np: {chunks_np} ...")  # 确认是['str1', 'str2', ...]
            logging.warning(f"第一个模型输出数组形状: {chunks_np.shape}")  # 例如(3,)表示3个元素

            # 4. 创建输出张量
            # 对于字符串张量，numpy数组的dtype需要是object
            out_tensor = pb_utils.Tensor(
                "OUTPUT_CHUNKS",
                chunks_np  # 直接使用一维数组，无需额外reshape
            )

            # 5. 创建并发送响应
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(inference_response)

        return responses

    def finalize(self):
        """
        模型卸载时调用。
        """
        print('Text Splitter cleaning up...')