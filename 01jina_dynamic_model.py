import logging
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from towhee.serve.triton.bls.python_backend_wrapper import pb_utils



class JinaEmbeddingTriton:
    def __init__(self):
        self.model_path = None
        self.device = "cpu"
        self.model = None
        self.tokenizer = None
        # 初始化时不加载模型，等待Triton的initialize信号

    def initialize(self, args):
        """Triton初始化方法，从配置中获取模型路径和设备信息"""
        self.model_path = Path(args["model_repository"] + "/" + args["model_version"])
        self.device = "cuda" if args.get("device", "cpu") == "gpu" else "cpu"
        self._load_model()
        logging.warning(f"Jina模型初始化完成，路径: {self.model_path}, 设备: {self.device}")

    def _load_model(self):
        """加载Jina模型和分词器"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型路径不存在: {self.model_path}")

        # 加载分词器和模型
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
            device_map=self.device
        )
        self.model.eval()  # 推理模式

    def execute(self, requests):
        """处理批量请求的主方法"""
        logging.warning(f'开始处理批量请求，共{len(requests)}个请求')

        # 1. 收集所有请求中的文本和task_id，合并成一个大批次
        all_texts = []
        # all_task_ids = []
        request_sizes = []  # 记录每个请求包含的样本数，用于后续拆分结果

        for request in requests:
            # 获取当前请求的输入文本
            text_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT")
            text_np = text_tensor.as_numpy()  # 形状为[batch_size, 1]

            # 获取当前请求的task_id
            # task_id_tensor = pb_utils.get_input_tensor_by_name(request, "TASK_ID")
            # task_id_np = task_id_tensor.as_numpy()  # 形状为[batch_size, 1]或[1]

            batch_size = text_np.shape[0]
            request_sizes.append(batch_size)

            # 解析当前请求的所有文本和task_id
            for i in range(batch_size):
                # 处理文本
                text = text_np[i][0].decode('utf-8') if isinstance(text_np[i][0], bytes) else str(text_np[i][0])
                all_texts.append(text)

                # 处理task_id，如果整个批次共享一个task_id则取第一个值
                # if task_id_np.ndim == 1:
                #     task_id = task_id_np[0]
                # else:
                #     task_id = task_id_np[i][0]
                # all_task_ids.append(task_id)

        logging.warning(f'合并后的总样本数: {len(all_texts)}')

        # 2. 对合并后的所有文本进行统一预处理和推理（批量处理）
        if not all_texts:
            # 空请求处理
            responses = []
            for _ in requests:
                response = pb_utils.InferenceResponse()
                responses.append(response)
            return responses

        # 处理输入文本
        batch_inputs = self.tokenizer(
            all_texts,
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt",
            return_attention_mask=True
        )

        # 转移到目标设备
        input_ids = batch_inputs['input_ids'].to(self.device)
        attention_mask = batch_inputs['attention_mask'].to(self.device)
        num_examples = len(all_texts)

        # 使用输入的task_id创建适配器掩码
        task_id = 1
        adapter_mask = torch.full(
            (num_examples,), task_id, dtype=torch.int32, device=device
        )
        logging.warning('------jina执行了一次推理！-------')

        # 模型推理
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                adapter_mask=adapter_mask,
                return_dict=True
            )
            token_embeddings = outputs.last_hidden_state

        # 扩展掩码维度以匹配隐藏层维度
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        # 加权求和后除以有效token数量 → (batch_size, hidden_dim)
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        valid_tokens = torch.clamp(mask_expanded.sum(1), min=1e-9)
        sentence_embeddings = sum_embeddings / valid_tokens
        # 转换为numpy数组
        embeddings_np = sentence_embeddings.cpu().numpy()

        # 3. 将结果拆分回各个请求
        responses = []
        current_idx = 0

        for req_size in request_sizes:
            # 提取当前请求的结果
            req_embeddings = embeddings_np[current_idx:current_idx + req_size]
            current_idx += req_size

            # 创建输出张量
            output_tensor = pb_utils.Tensor("EMBEDDINGS", req_embeddings)

            # 创建响应
            inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(inference_response)

        logging.warning(f'批量请求处理完成，共返回{len(responses)}个响应')
        return responses
