#coding=utf-8
from transformers import AutoTokenizer
from towhee.serve.triton.bls.python_backend_wrapper import pb_utils
import numpy as np
import logging
logging.basicConfig(filename='/workspace/log_res/onnx.log', level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

class TritonPythonModel:
    def initialize(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained("/workspace/models/bge-large-onnx")

    def execute(self, requests):
        logging.warning(f'开始处理批量请求，共{len(requests)}个请求')

        # 1. 收集所有请求中的文本，合并成一个大批次
        all_texts = []
        request_sizes = []  # 记录每个请求包含的样本数，用于后续拆分结果

        for request in requests:
            # 获取当前请求的输入文本
            text_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT")
            text_np = text_tensor.as_numpy()  # 形状为[batch_size, 1]

            batch_size = text_np.shape[0]
            request_sizes.append(batch_size)

            # 解析当前请求的所有文本
            for i in range(batch_size):
                text = text_np[i][0].decode('utf-8') if isinstance(text_np[i][0], bytes) else str(text_np[i][0])
                all_texts.append(text)

        logging.warning(f'合并后的总样本数: {len(all_texts)}')

        # 2. 对合并后的所有文本进行统一预处理（批量处理）
        inputs = self.tokenizer(
            all_texts,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=512
        )
        logging.warning(f'批量预处理后的input_ids形状: {inputs["input_ids"].shape}')

        # 3. 调用ONNX模型（一次调用处理整个批次）
        infer_request = pb_utils.InferenceRequest(
            model_name="bge_embedding",
            inputs=[
                pb_utils.Tensor("input_ids", inputs['input_ids']),
                pb_utils.Tensor("attention_mask", inputs['attention_mask']),
                pb_utils.Tensor("token_type_ids", inputs['token_type_ids'])
            ],
            requested_output_names=["last_hidden_state"]
        )
        infer_response = infer_request.exec()

        # 4. 批量后处理（Pooling）
        hidden_state = pb_utils.get_output_tensor_by_name(infer_response, "last_hidden_state").as_numpy()
        logging.warning(f'模型输出的hidden_state形状: {hidden_state.shape}')

        embeddings = self.pooling(hidden_state, inputs['attention_mask'])
        logging.warning(f'池化后的embeddings形状: {embeddings.shape}')

        # 5. 拆分结果，为每个原始请求生成对应的响应
        responses = []
        current_idx = 0

        for req_size in request_sizes:
            # 提取当前请求对应的embeddings
            req_embeddings = embeddings[current_idx:current_idx + req_size]
            current_idx += req_size

            # 创建输出张量并添加到响应列表
            output_tensor = pb_utils.Tensor("EMBEDDING", req_embeddings.astype(np.float32))
            responses.append(pb_utils.InferenceResponse([output_tensor]))

        return responses

    def pooling(self, hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """
        参数:
            hidden_state: 模型输出的隐藏状态，形状 [batch_size, seq_len, hidden_dim]
            attention_mask: 注意力掩码，形状 [batch_size, seq_len]
        返回:
            池化后的句子嵌入向量，形状 [batch_size, hidden_dim]
        """
        # 扩展注意力掩码维度：[batch_size, seq_len] → [batch_size, seq_len, 1]
        expanded_mask = np.expand_dims(attention_mask, axis=-1)
        # 用掩码过滤无效Token（填充部分）
        masked_hidden = hidden_state * expanded_mask
        # 计算有效Token的加权和（沿序列维度求和）
        sum_embeddings = np.sum(masked_hidden, axis=1)
        # 计算每个样本的有效Token数量（避免除零）
        valid_tokens = np.maximum(np.sum(attention_mask, axis=1, keepdims=True), 1e-9)
        # 计算均值池化结果
        sentence_embeddings = sum_embeddings / valid_tokens
        # 可选：对嵌入向量进行L2归一化（增强相似性计算效果）
        normalized_embeddings = sentence_embeddings / np.linalg.norm(
            sentence_embeddings, axis=1, keepdims=True
        )
        return normalized_embeddings

# ----------------------批量推理--------------------------------------------------------------------------
import requests
import json
import numpy as np
def batch_embedding_via_http(texts, model_name, url="localhost:8000"):
    batch_size = len(texts)
    if batch_size == 0:
        return None

    input_data = {
        "inputs": [
            {
                "name": "TEXT",
                "shape": [batch_size, 1],
                "datatype": "BYTES",
                "data": texts
            }
        ],
        "outputs": [
            {
                "name": "EMBEDDING",
                "parameters": {"binary_data": False}
            }
        ]
    }

    try:
        infer_url = f"http://{url}/v2/models/{model_name}/infer"
        response = requests.post(
            infer_url,
            json=input_data,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code != 200:
            print(f"❌ 请求失败，状态码: {response.status_code}")
            print(f"错误详情: {response.text}")
            return None

        result = response.json()
        embeddings_flat = result["outputs"][0]["data"]

        embedding_dim = len(embeddings_flat) // batch_size
        if len(embeddings_flat) % batch_size != 0:
            print(f"⚠️ 嵌入向量长度不匹配，总长度: {len(embeddings_flat)}, 批量大小: {batch_size}")
            return None

        return np.array(embeddings_flat, dtype=np.float32).reshape(batch_size, embedding_dim)

    except Exception as e:
        print(f"❌ 批量调用错误: {str(e)}")
        return None

import time
if __name__ == "__main__":
    triton_url = "localhost:8000"
    model_name = "bge_pipeline"

    test_texts = [f"这是第{i + 1}条测试文本" for i in range(16)]

    print(f"批量测试文本数量: {len(test_texts)}")
    start_time = time.time()

    batch_embeddings = batch_embedding_via_http(test_texts, model_name, triton_url)

    end_time = time.time()  # 记录函数结束执行的时间
    elapsed_time = end_time - start_time  # 计算函数执行的耗时
    print(f"Function  took {elapsed_time:.6f} seconds to execute")
    if batch_embeddings is not None:
        print(f"批量嵌入结果形状: {batch_embeddings.shape}")
        for i in range(min(3, len(test_texts))):
            print(f"\n文本 {i + 1}: {test_texts[i]}")
            print(f"嵌入向量前5个值: {batch_embeddings[i, :9]}")

# ------------------------------异步推理--------------------------------------------
import asyncio
import aiohttp
import numpy as np
import json
async def async_infer(session, text, model_name, triton_url):
    """单个文本的异步推理请求"""
    request_data = {
        "inputs": [
            {
                "name": "TEXT",
                "shape": [1, 1],  # 保持批次维度，让Triton可以自动组合批次
                "datatype": "BYTES",
                "data": [text]
            }
        ],
        "outputs": [
            {
                "name": "EMBEDDING",
                "parameters": {"binary_data": False}
            }
        ]
    }

    try:
        async with session.post(f"{triton_url}/v2/models/{model_name}/infer",
                                json=request_data) as response:
            if response.status == 200:
                result = await response.json()
                return np.array(result["outputs"][0]["data"], dtype=np.float32)
            else:
                content = await response.text()
                print(f"请求失败: {response.status}, 文本: {text}, 错误: {content}")
                return None
    except Exception as e:
        print(f"请求错误: {str(e)}, 文本: {text}")
        return None


async def main():
    # 配置参数
    triton_url = "http://localhost:8000"
    model_name = "bge_pipeline"
    test_texts = [
        f"这是第{i + 1}条测试文本" for i in range(10)
    ]

    # 并发发送所有请求，让Triton自动组合批次
    async with aiohttp.ClientSession() as session:
        # 创建所有请求任务
        tasks = [
            async_infer(session, text, model_name, triton_url)
            for text in test_texts
        ]

        # 等待所有请求完成
        results = await asyncio.gather(*tasks)

    # 打印结果
    for i, (text, embedding) in enumerate(zip(test_texts, results)):
        if embedding is not None:
            print(f"文本 {i + 1}: {text}")
            # print(f"嵌入向量长度: {len(embedding)}")
            print(f"嵌入向量前5个值: {embedding[:8]}\n")

import time
if __name__ == "__main__":
    start_time = time.time()
    asyncio.run(main())
    end_time = time.time()  # 记录函数结束执行的时间
    elapsed_time = end_time - start_time  # 计算函数执行的耗时
    print(f"Function  took {elapsed_time:.6f} seconds to execute")

# ----------------------------单次推理----------------------------------------
import numpy as np
import requests
import json
import time

def embedding_via_http(text, model_name, url="localhost:8000"):
    # 1. 对输入文本进行JSON序列化

    input_data = {
        "inputs": [
            {
                "name": "TEXT",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [text]
            }
        ],
        "outputs": [
            {
                "name": "EMBEDDING",
                "parameters": {"binary_data": False}
            }
        ]
    }

    try:
        infer_url = f"http://{url}/v2/models/{model_name}/infer"
        response = requests.post(
            infer_url,
            json=input_data,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code != 200:
            print(f"❌ 请求失败，状态码: {response.status_code}")
            print(f"错误详情: {response.text}")
            return None

        # 2. 解析响应结果（关键修改）
        result = response.json()
        # 使用Towhee的from_json工具解析特殊格式的向量
        result = result["outputs"][0]["data"]
        return result

    except Exception as e:
        print(f"❌ 调用错误: {str(e)}")
        return None
if __name__ == "__main__":
    triton_url = "localhost:8000"
    model_name = "bge_pipeline"
    test_text = "这是一个测试句子"

    print(f"测试文本: {test_text}")
    embedding = embedding_via_http(test_text, model_name, triton_url)

    if embedding is not None:
        print(f"向量前5值: {embedding[:5]}")