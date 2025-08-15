import numpy as np
import tritonclient.http as httpclient

def test_embedding_pipeline():
    """
    测试端到端的文本嵌入流水线。
    """
    # 准备输入数据：一个长字符串
    long_text = "Triton推理服务器是NVIDIA开发的一款高性能推理解决方案。" \
                "它支持多种深度学习和机器学习框架，如TensorFlow、PyTorch、TensorRT和ONNX。" \
                "通过Triton，开发者可以轻松地将训练好的模型部署为生产级服务。" \
                "它提供了HTTP和gRPC两种协议接口，并支持动态批处理、模型版本控制和多模型集成等高级功能。" \
                "这使得Triton成为构建可扩展、高吞吐量AI应用的理想选择。工业上已经广泛应用这种方式部署模型。" \
                "此文章面向的读者对象为刚接触LMS的同学，意在为加入LMS团队的新同学提供一个手把手教程来搭建LMS开发测试环境，为之后的开发打下基础。在搭建的过程中，可以同时了解到一些Docker、K8s、Linux的相关基础知识。"

    # Triton的字符串输入需要是numpy数组，类型为object
    input_text_np = np.array([long_text.encode('utf-8')], dtype=object)

    # 创建Triton客户端
    try:
        triton_client = httpclient.InferenceServerClient(url="localhost:8000", verbose=False)
    except Exception as e:
        print(f"无法连接到Triton服务器: {e}")
        return

    # 准备输入张量
    inputs = [
        httpclient.InferInput("PIPELINE_INPUT", [1], "BYTES"),
    ]
    inputs[0].set_data_from_numpy(input_text_np)

    # 准备输出张量
    outputs = [
        httpclient.InferRequestedOutput("PIPELINE_OUTPUT"),
    ]

    print("向Triton服务器发送请求...")
    # 发送推理请求到ensemble模型
    results = triton_client.infer(
        model_name="text_embedding_pipeline",
        inputs=inputs,
        outputs=outputs
    )

    # 获取结果
    embedding_vectors = results.as_numpy("PIPELINE_OUTPUT")

    print("成功接收到响应！")
    print(f"输入文本切分后的块数: {embedding_vectors.shape[0]}")
    print(f"每个嵌入向量的维度: {embedding_vectors.shape[1]}")
    print(f"最终输出向量的形状: {embedding_vectors.shape}")
    print("\n前两个嵌入向量（部分）:")
    print(embedding_vectors[:3, :5])

if __name__ == '__main__':
    test_embedding_pipeline()
