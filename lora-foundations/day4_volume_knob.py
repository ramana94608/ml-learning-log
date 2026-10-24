import numpy as np

x = np.array([1.0, 2.0])        # one input reading
W = np.array([[1.0, 0.0],
              [0.0, 1.0]])      # the frozen scientist
A = np.array([[1.0, 1.0]])      # sticky note, sheet 1: one row
B = np.array([[1.0],
              [0.0]])           # sticky note, sheet 2: one column

nudge = B @ A @ x

print("frozen:", W @ x)
print("nudge:", nudge)
for s in [0, 0.5, 1, 2]:
    print(s, W @ x + s * nudge)
