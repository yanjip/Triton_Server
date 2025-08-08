import requests
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor

# 全局变量用于记录总请求数
total_requests = 0
completed_requests = 0
lock = threading.Lock()


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        global completed_requests
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time

        # 线程安全地更新完成的请求数
        with lock:
            completed_requests += 1
            print(f"请求 {completed_requests}/{total_requests} 完成，耗时 {elapsed_time:.6f} 秒")

        return result

    return wrapper


@timing_decorator
def embedding_via_http(text, model_name, url="localhost:8000"):
    # 对输入文本进行JSON序列化
    serialized_text = json.dumps(text)

    input_data = {
        "inputs": [
            {
                "name": "INPUT0",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [serialized_text]
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

        # 解析响应结果
        result = response.json()
        embedding_str = result["outputs"][0]["data"][0]
        parsed_data = json.loads(embedding_str)
        result = parsed_data[0][0][0]['data']
        return result

    except Exception as e:
        print(f"❌ 调用错误: {str(e)}")
        return None


def client_task(client_id, model_name, triton_url, requests_per_client):
    """单个客户端的任务：发送指定次数的请求"""
    print(f"客户端 {client_id} 开始发送 {requests_per_client} 次请求...")
    for i in range(requests_per_client):
        # 每次请求使用不同的文本（模拟实际场景）
        text = [f"客户端 {client_id} 的第 {i + 1} 条请求: Hello World {i}"]
        embedding = embedding_via_http(text, model_name, triton_url)
        if embedding:
            print(f"客户端 {client_id} 第 {i + 1} 次请求成功，向量前5值: {embedding[0][:5]}")


if __name__ == "__main__":
    triton_url = "localhost:8000"
    model_name = "pipeline"
    num_clients = 8  # 客户端数量
    requests_per_client = 4  # 每个客户端发送的请求数
    total_requests = num_clients * requests_per_client

    print(f"===== 开始测试：{num_clients} 个客户端，每个发送 {requests_per_client} 次请求 =====")
    start_time = time.time()

    # 使用线程池模拟多个客户端
    with ThreadPoolExecutor(max_workers=num_clients) as executor:
        # 为每个客户端提交一个任务
        for client_id in range(num_clients):
            executor.submit(
                client_task,
                client_id=client_id + 1,
                model_name=model_name,
                triton_url=triton_url,
                requests_per_client=requests_per_client
            )

    total_time = time.time() - start_time
    print(f"\n===== 测试完成 =====")
    print(f"总请求数: {total_requests}")
    print(f"总耗时: {total_time:.6f} 秒")
    print(f"平均请求耗时: {total_time / total_requests:.6f} 秒")
