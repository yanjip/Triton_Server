from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from towhee import register, pipe, ops, AutoConfig, build_pipeline_model
from towhee.operator import PyOperator

@register(name="bge_embedding")
class BGEEmbedding(PyOperator):
    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = Path(model_path)
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型路径不存在: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = AutoModel.from_pretrained(str(self.model_path), device_map="cpu")
        self.model.eval()

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state if hasattr(model_output, "last_hidden_state") else model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        counts = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return summed / counts

    def __call__(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, padding="longest", truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            out = self.model(**inputs)
        emb = self._mean_pooling(out, inputs["attention_mask"])
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)[0].cpu().numpy()
        return emb

def build(triton_model_root: str, cpus_per_instance: int, max_batch: int, latency_us: int, preferred_batches: list):
    config = AutoConfig.TritonCPUConfig(
        num_instances_per_device=cpus_per_instance,
        max_batch_size=max_batch,
        batch_latency_micros=latency_us,
        preferred_batch_size=preferred_batches
    )

    p = (
        pipe.input("text")
        .map("text", "embedding", ops.bge_embedding(model_path="/opt/embedding/model/bge-large-zh-v1.5"),
             config=config)
        .output("embedding")
    )

    print(">> 调用 Towhee 构建 Triton 模型...")
    build_pipeline_model(
        dc_pipeline=p,
        model_root=triton_model_root,
        format_priority=["onnx"],
        parallelism=cpus_per_instance,
        server="triton"
    )

    print(f">> 构建完成，模型目录：{triton_model_root}")
def deploy():
    triton_model_root = "/workspace/triton_config"
    cpus = 2
    max_batch = 32
    latency_us = 500000
    preferred_batches = [16, 32]
    build(triton_model_root, cpus, max_batch, latency_us, preferred_batches)
if __name__ == '__main__':
    deploy()

# ------------------------批量推理------------------------------------------
import numpy as np
import requests
import json

import time

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录函数开始执行的时间
        result = func(*args, **kwargs)  # 执行函数
        end_time = time.time()  # 记录函数结束执行的时间
        elapsed_time = end_time - start_time  # 计算函数执行的耗时
        print(f"Function {func.__name__} took {elapsed_time:.6f} seconds to execute")
        return result
    return wrapper

# 示例函数
@timing_decorator
def batch_embedding_via_http(texts, model_name, url="localhost:8000"):
    """
    texts: 文本列表，如["文本1", "文本2", ..., "文本N"]
    """
    # 1. 验证输入批量大小不超过模型最大支持的batch_size（8）
    batch_size = len(texts)

    # 2. 对每条文本进行JSON序列化
    serialized_texts = [json.dumps(text) for text in texts]

    # 3. 构造批量请求（关键：shape为[batch_size, 1]）
    input_data = {
        "inputs": [
            {
                "name": "INPUT0",
                "shape": [batch_size, 1],  # 第一维为批量大小，第二维为单条数据维度
                "datatype": "BYTES",
                "data": serialized_texts  # 传入批量文本列表
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

        # 4. 解析批量结果
        result = response.json()
        embedding_strs = result["outputs"][0]["data"][0]  # 批量结果列表
        return embedding_strs  # 返回形状为[batch_size, 向量维度]的数组
    except Exception as e:
        print(f"❌ 批量调用错误: {str(e)}")
        return None
if __name__ == "__main__":
    triton_url = "localhost:8000"
    model_name = "pipeline"
    test_texts = [
        "这是第一条测试文本",
        "这是第一条测试文本",
    ]
    print(f"批量测试文本数量: {len(test_texts)}")
    batch_embeddings = batch_embedding_via_http(test_texts, model_name, triton_url)
    print(f"向量前{len(test_texts)}值: {batch_embeddings[:105]}")

# --------------------------异步推理----------------------------------------
import json
import time
import asyncio
import aiohttp
async def async_infer(session, text, model_name, url):
    """异步发送单个推理请求"""
    # 序列化输入文本
    input_data = json.dumps(text)

    # 构造请求体
    payload = {
        "inputs": [
            {
                "name": "INPUT0",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [input_data]
            }
        ]
    }

    # 发送异步POST请求
    async with session.post(
            f"http://{url}/v2/models/{model_name}/infer",
            json=payload,
            headers={"Content-Type": "application/json"}
    ) as response:
        if response.status != 200:
            error = await response.text()
            return f"请求失败: {error}"

        result = await response.json()
        return result["outputs"][0]["data"][0]

async def main():
    model_name = "pipeline"
    triton_url = "localhost:8000"  # Triton服务器地址

    # 测试文本列表（模拟多个并发请求）
    test_texts = [
        "这是第一条测试文本",
        "这是第二条测试文本",
    ]
    print(f"批量测试文本数量: {len(test_texts)}")
    # 创建异步会话
    async with aiohttp.ClientSession() as session:
        # 并发发送所有请求（关键：同时触发多个请求，让Triton合并为批次）
        tasks = [
            async_infer(session, text, model_name, triton_url)
            for text in test_texts
        ]
        # 等待所有请求完成
        results = await asyncio.gather(*tasks)
    for i, res in enumerate(results):
        print(f"前5个值: {res[:105]}\n")

if __name__ == "__main__":
    # 运行异步事件循环
    start_time = time.time()
    asyncio.run(main())
    end_time = time.time()  # 记录函数结束执行的时间
    elapsed_time = end_time - start_time  # 计算函数执行的耗时
    print(f"Function  took {elapsed_time:.6f} seconds to execute")