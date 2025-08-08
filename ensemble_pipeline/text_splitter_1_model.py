import json
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """
    一个简单的文本切分模型。
    输入一个长字符串，按固定长度切分成一个字符串列表。
    """

    def initialize(self, args):
        """
        模型初始化时调用。
        """
        self.model_config = json.loads(args['model_config'])
        print("Text Splitter model initialized")

    def execute(self, requests):
        """
        处理推理请求。
        """
        responses = []

        # 遍历所有请求
        for request in requests:
            # 1. 获取输入张量
            in_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT")

            # 2. 将张量数据解码为UTF-8字符串
            # 输入shape为[1]，所以我们取第一个元素
            long_text = in_tensor.as_numpy()[0].decode('utf-8')

            # 3. 执行切分逻辑（这里使用简单的固定长度切分作为示例）
            chunk_size = 200  # 定义每个切片的最大长度
            chunks = [long_text[i:i + chunk_size] for i in range(0, len(long_text), chunk_size)]

            # 4. 创建输出张量
            # 对于字符串张量，numpy数组的dtype需要是object
            out_tensor = pb_utils.Tensor(
                "OUTPUT_CHUNKS",
                np.array(chunks, dtype=object)
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
