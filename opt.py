import numpy as np
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig
from sim import do_simulation

# ---------- 1. 定义搜索空间 ----------
param_configs = [
    RangeParameterConfig(          # 连续区间参数
        name=f"x{i}",              # 对应 datas[:, i]
        bounds=(0.1, 0.9),
        parameter_type="float",
    )
    for i in range(7)
]

# ---------- 2. 初始化 Client ----------
name = "heat_sink_optimization_8_symmetric"
client = Client()
client.configure_experiment(
    name=name,
    parameters=param_configs,
)

client.configure_optimization(objective="score")
# client = Client.load_from_json_file(f"{name}.json")

# ---------- 3. 循环 ask-tell ----------
# 这里一次取 16 个 trial，恰好塞满你的并行仿真
batch_size = 16
iters = 32          # 6 批 × 16，可按预算调整

for _ in range(iters):
    trial_params = client.get_next_trials(max_trials=batch_size)  # ask
    
    batch = []
    for p in trial_params.values():
        x = [p[f"x{i}"] for i in range(7)]
        batch.append(np.array(x))
    
    scores = do_simulation(batch)
    # tell
    for (trial_idx, _), s in zip(trial_params.items(), scores):
        client.complete_trial(trial_index=trial_idx,
                              raw_data={"score": float(s)})
    client.save_to_json_file(filepath=f"{name}.json")

# ---------- 4. 读取结果 ----------
best_parameters, prediction, index, name = client.get_best_parameterization()
print("Best Parameters:", best_parameters)
print("Prediction (mean, variance):", prediction)