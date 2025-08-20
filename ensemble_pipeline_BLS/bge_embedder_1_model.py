#coding=utf-8
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

    def execute(self, requests):
        logging.warning(f'开始处理批量请求，共{len(requests)}个请求')

        responses = []
        for request in requests:
            # 获取当前请求的输入文本
            in_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_CHUNKS")
            in_np = in_tensor.as_numpy()
            logging.warning(f"shape of in_np: {in_np.shape}")
            texts = [s.decode('utf-8') for s in in_np]
            logging.warning(f" len(texts): {len(texts)}")
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
                emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

            responses.append(pb_utils.InferenceResponse([
                pb_utils.Tensor("OUTPUT_EMBEDDINGS", emb.astype(np.float32))
            ]))

        return responses