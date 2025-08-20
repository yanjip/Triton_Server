import json
import os
import asyncio
import numpy as np
from towhee.serve.triton.bls.python_backend_wrapper import pb_utils
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


    async def execute(self, requests):
        responses = []
        # 遍历所有请求
        for request in requests:
            # 1. 获取输入张量
            in_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT")
            # 2. 将张量数据解码为UTF-8字符串，输入shape为[1]，所以我们取第一个元素
            long_text = in_tensor.as_numpy()[0].decode('utf-8')

            # 3. 执行切分逻辑（这里使用简单的固定长度切分作为示例）
            chunk_size = 15  # 定义每个切片的最大长度
            chunks = [long_text[i:i + chunk_size] for i in range(0, len(long_text), chunk_size)]
            logging.warning(f"切分后的原始列表: {chunks[:4]} ...")  # 确认是['str1', 'str2', ...]

            # 按 max_batch_size=8 分批
            max_b = 8
            futures = []
            for i in range(0, len(chunks), max_b):
                batch = chunks[i:i + max_b]
                t_in = pb_utils.Tensor("INPUT_CHUNKS", np.array(batch, dtype=object))
                req = pb_utils.InferenceRequest(
                    model_name="bge_embedder",
                    requested_output_names=["OUTPUT_EMBEDDINGS"],
                    inputs=[t_in]
                )
                futures.append(req.async_exec())
                # 等待所有 embed 请求完成

            embed_resps = await asyncio.gather(*futures)
            logging.warning("shape of embed_resps: {}".format(len(embed_resps)))
            all_embs = []
            for er in embed_resps:
                if er.has_error():
                    raise pb_utils.TritonModelException(er.error().message())
                ot = pb_utils.get_output_tensor_by_name(er, "OUTPUT_EMBEDDINGS")
                all_embs.append(ot.as_numpy())

            combined = np.vstack(all_embs).astype(np.float32)
            out = pb_utils.Tensor("FINAL_EMBED", combined)
            responses.append(pb_utils.InferenceResponse([out]))
        return responses

    def finalize(self):
        """
        模型卸载时调用。
        """
        print('Text Splitter cleaning up...')
