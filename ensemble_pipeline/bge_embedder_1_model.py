import json
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """
    使用Hugging Face Transformers加载本地BGE模型进行文本嵌入。
    """

    def initialize(self, args):
        """
        在模型加载时初始化分词器和模型。
        """
        self.model_config = json.loads(args['model_config'])

        # 确定设备 (GPU or CPU)
        instance_group_kind = pb_utils.get_instance_group_kind(args)
        if instance_group_kind == pb_utils.TRITONSERVER_INSTANCEGROUPKIND_GPU:
            self.device = "cuda"
        else:
            self.device = "cpu"

        # 获取模型路径 (模型文件存放在当前脚本所在目录的子目录中)
        # 例如: bge_embedder/1/bge-base-zh-v1.5/
        model_dir = os.path.dirname(__file__)
        # 你需要将HuggingFace下载的模型文件夹名称填在这里
        local_model_path = os.path.join(model_dir, "bge-base-zh-v1.5")

        if not os.path.isdir(local_model_path):
            raise pb_utils.TritonModelException(f"Model directory not found at: {local_model_path}")

        print(f"Loading model from {local_model_path} to {self.device}...")

        # 加载分词器和模型
        self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        self.model = AutoModel.from_pretrained(local_model_path)
        self.model.to(self.device)
        self.model.eval()

        print("BGE Embedder model initialized successfully.")

    def execute(self, requests):
        """
        处理推理请求。Triton会自动将请求聚合成批次。
        """
        responses = []

        for request in requests:
            # 1. 获取输入张量
            in_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_CHUNKS")

            # 2. 将输入张量解码为字符串列表
            # .as_numpy() 返回一个包含bytes的numpy数组
            text_list = [t.decode('utf-8') for t in in_tensor.as_numpy()]

            # 3. 使用分词器处理文本
            encoded_input = self.tokenizer(
                text_list,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(self.device)

            # 4. 执行模型推理
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                # 取[CLS] token的嵌入
                sentence_embeddings = model_output[0][:, 0]

            # 5. L2归一化 (BGE模型的标准做法)
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

            # 6. 创建输出张量
            out_tensor = pb_utils.Tensor(
                "OUTPUT_EMBEDDINGS",
                sentence_embeddings.cpu().numpy()  # 将结果移回CPU
            )

            # 7. 创建并发送响应
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(inference_response)

        return responses

    def finalize(self):
        """
        模型卸载时调用。
        """
        print('BGE Embedder cleaning up...')
        self.model = None
        self.tokenizer = None
