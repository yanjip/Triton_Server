import psutil
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

# 初始化数据
timestamps, cpu_totals, memory_usages = [], [], []
duration = 300  # 监控5分钟
interval = 0.5  # 每0.5秒采样一次

print("开始监控CPU和内存（按Ctrl+C停止）...")
try:
    for i in range(int(duration / interval)):
        # 获取所有核心的使用率并求和（多核总使用率）
        per_cpu = psutil.cpu_percent(interval=interval, percpu=True)
        total_usage = sum(per_cpu)  # 多核总使用率 = 各核心之和

        # 获取内存使用率
        memory = psutil.virtual_memory()
        memory_usage = memory.percent  # 内存使用率（百分比）

        # 记录数据
        timestamps.append(datetime.now().strftime("%H:%M:%S"))
        cpu_totals.append(total_usage)
        memory_usages.append(memory_usage)

        # 每10个点打印一次，避免输出过多
        if i % 10 == 0:
            print(f"{timestamps[-1]} - CPU总使用率: {total_usage:.1f}% - 内存使用率: {memory_usage:.1f}%")

except KeyboardInterrupt:
    print("\n监控已手动停止")

# 创建图表和主轴（CPU）
fig, ax1 = plt.subplots(figsize=(14, 7))

# 使用索引作为x值，而不是直接使用时间字符串，解决刻度分布问题
x = np.arange(len(timestamps))

# 绘制CPU使用率（左轴）
color = '#FF6B6B'
ax1.set_xlabel('Time', fontsize=12)
ax1.set_ylabel('CPU Usage (%)', color=color, fontsize=12)
ax1.plot(x, cpu_totals, marker='o', linestyle='-', color=color, label='CPU Usage')
ax1.fill_between(x, cpu_totals, alpha=0.2, color=color)
ax1.tick_params(axis='y', labelcolor=color)

# 设置CPU轴的合理范围
max_cpu = max(cpu_totals) if cpu_totals else 0
core_count = len(per_cpu) if 'per_cpu' in locals() else 1
ax1.set_ylim(0, min(100 * core_count, max_cpu * 1.1))

# 创建副轴（内存）
ax2 = ax1.twinx()  # 共享x轴
color = '#4ECDC4'
ax2.set_ylabel('Memory Usage (%)', color=color, fontsize=12)
ax2.plot(x, memory_usages, marker='s', linestyle='-', color=color, label='Memory Usage')
ax2.fill_between(x, memory_usages, alpha=0.2, color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, 100)  # 内存使用率范围固定0-100%

# 优化X轴刻度显示
plt.title(f'CPU and Memory Usage ({core_count} Cores)', fontsize=14)

# 计算需要显示的刻度数量（根据数据量动态调整）
num_ticks = min(10, len(timestamps))  # 最多显示10个刻度
if num_ticks > 0:
    tick_interval = max(1, len(timestamps) // num_ticks)
    tick_positions = np.arange(0, len(timestamps), tick_interval)
    tick_labels = [timestamps[i] for i in tick_positions]

    # 设置X轴刻度和标签
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels, rotation=45)

ax1.grid(True, linestyle='--', alpha=0.7)

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()

# 保存图表
plt.savefig(f"cpu_memory_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", dpi=300)
plt.show()
