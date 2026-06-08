import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(0)

model = nn.Sequential(nn.Linear(8, 8), nn.Dropout(p=0.5), nn.Linear(8, 8))
model.train()

x = torch.ones(2, 8)  # 2 queries, dim=8

# Method 1: T separate forward passes (loop)
passes_loop = []
with torch.no_grad():
    for t in range(3):
        out = model(x)
        passes_loop.append(out.numpy())
loop_result = np.stack(passes_loop, axis=0)  # (3, 2, 8)

torch.manual_seed(0)
model2 = nn.Sequential(nn.Linear(8, 8), nn.Dropout(p=0.5), nn.Linear(8, 8))
model2.load_state_dict(model.state_dict())
model2.train()

# Method 2: vectorized (what run_grass.py does)
with torch.no_grad():
    out = model2(x.repeat(3, 1))  # (6, 8)
vec_result = out.numpy().reshape(3, 2, 8)  # (3, 2, 8)

print("=== LOOP METHOD ===")
for t in range(3):
    for q in range(2):
        print(f"pass {t+1}, query {q}: {loop_result[t, q]}")
    print()

print("=== VECTORIZED METHOD ===")
for t in range(3):
    for q in range(2):
        print(f"pass {t+1}, query {q}: {vec_result[t, q]}")
    print()

print("=== EQUIVALENCE CHECK ===")
print(f"Results identical:  {np.allclose(loop_result, vec_result)}")
print(f"Means identical:    {np.allclose(loop_result.mean(0), vec_result.mean(0))}")
print(f"Stds identical:     {np.allclose(loop_result.std(0),  vec_result.std(0))}")
print(f"Loop sigma mean:    {loop_result.std(axis=0).mean():.6f}")
print(f"Vec  sigma mean:    {vec_result.std(axis=0).mean():.6f}")
