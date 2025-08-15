import torch
import os
from transformers import AutoTokenizer, AutoModel
from towhee.serve.triton.bls.python_backend_wrapper import pb_utils
import numpy as np
from tritonclient.grpc import model_config_pb2
from google.protobuf import text_format

import logging
logging.basicConfig(filename='/workspace/log_res/onnx.log', level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
class TritonPythonModel:
    def initialize(self, args):
        local_model_path = "/opt/embedding/model/bge-large-zh-v1.5"
        self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        self.model = AutoModel.from_pretrained(local_model_path)
        self.device = 'cpu'
        self.model.to()
        self.model.eval()

        self.model_name = args['model_name']  # 自身模型名称
        self.model_repo = args['model_repository']  # 模型仓库路径
        self.max_batch_size = 1  # 默认批次大小

        # 读取自身配置文件获取max_batch_size
        try:
            config_path = os.path.join(
                self.model_repo,
                "config.pbtxt"
            )
            with open(config_path, 'r') as f:
                config_text = f.read()

            model_config = model_config_pb2.ModelConfig()
            text_format.Merge(config_text, model_config)

            # 若配置中max_batch_size为0则使用1（表示不支持批处理）
            self.max_batch_size = model_config.max_batch_size or 1
            logging.warning(f"模型{self.model_name}的max_batch_size: {self.max_batch_size}")
        except Exception as e:
            logging.error(f"获取自身配置失败: {str(e)}，使用默认max_batch_size=1")


    def execute(self, requests):
        logging.warning(f'开始处理批量请求，共{len(requests)}个请求')

        # 1. 收集所有请求中的文本，合并成一个大批次
        all_texts = []
        request_sizes = []  # 记录每个请求包含的样本数，用于后续拆分结果

        for request in requests:
            # 获取当前请求的输入文本
            text_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_CHUNKS")
            text_np = text_tensor.as_numpy()  # 形状为[batch_size, 1]
            logging.warning(f'shape of text_np: {text_np.shape}')

            batch_size = text_np.shape[1]
            logging.warning(f'batch size of  text_np: {batch_size}')

            request_sizes.append(batch_size)
            for i in range(batch_size):
                # 解码字节串为字符串
                if isinstance(text_np[0][i], bytes):
                    text = text_np[0][i].decode('utf-8')
                else:
                    text = str(text_np[0][i])
                all_texts.append(text)
            logging.warning(f"解析后的字符串列表: {all_texts[:5]}")  # 应显示['str1', 'str2', ...]

        logging.warning(f'合并后的总样本数: {len(all_texts)}')

        # 2. 按max_batch_size拆分列表，分批推理
        all_embeddings = []
        total = len(all_texts)
        # 从0到total，按max_batch_size步长拆分
        for i in range(0, total, self.max_batch_size):
            batch_texts = all_texts[i:i + self.max_batch_size]
            logging.warning(f"处理子批次: {i // self.max_batch_size + 1}, 文本数量: {len(batch_texts)}")

            # 3. 批量预处理
            encoded_input = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            logging.warning(f'子批次input_ids形状: {encoded_input["input_ids"].shape}')

            # 4. 执行模型推理
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                sentence_embeddings = model_output[0][:, 0]  # 取[CLS] token

                # L2归一化
                sentence_embeddings = torch.nn.functional.normalize(
                    sentence_embeddings,
                    p=2,
                    dim=1
                )
                logging.warning(f'子批次embeddings形状: {sentence_embeddings.shape}')

                # 转换为numpy并添加到总结果
                all_embeddings.append(sentence_embeddings.cpu().numpy())

        # 5. 合并所有子批次结果
        if all_embeddings:
            sentence_embeddings = np.concatenate(all_embeddings, axis=0)
            logging.warning(f'合并后总embeddings形状: {sentence_embeddings.shape}')
        else:
            sentence_embeddings = np.array([])

        # 5. 拆分结果，为每个原始请求生成对应的响应
        responses = []
        current_idx = 0

        for req_size in request_sizes:
            # 提取当前请求对应的embeddings
            req_embeddings = sentence_embeddings[current_idx:current_idx + req_size]
            current_idx += req_size

            # 创建输出张量并添加到响应列表
            # 先将PyTorch张量转换为NumPy数组，再进行类型转换
            output_tensor = pb_utils.Tensor("OUTPUT_EMBEDDINGS",
                                            req_embeddings.astype(np.float32))
            responses.append(pb_utils.InferenceResponse([output_tensor]))

        return responses