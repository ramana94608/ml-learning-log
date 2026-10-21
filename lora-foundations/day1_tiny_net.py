"""
Day 1: what training actually is.
Run in the VS Code terminal: python3, then paste each block.

Data: sugar in (g/L) -> titer out (g/L).
The true relationship is roughly: titer = 0.5 * sugar.
"""
import numpy as np

sugar = np.array([10., 20., 30., 40., 50.])
titer = np.array([5.1, 9.8, 15.2, 20.1, 24.9])

# The "model": two knobs. w scales the input, b shifts the answer.
w = 0.0
b = 0.0

def predict(x):
    return w * x + b

def loss():
    # Loss = average wrongness, squared. Small loss means good guesses.
    guesses = predict(sugar)
    errors = guesses - titer
    return float(np.mean(errors ** 2))

print("start: w =", w, " b =", b, " loss =", round(loss(), 1))


# Training loop: for each step, find which direction makes the wrongness
# smaller, then nudge the knobs that way. lr = size of each nudge.
lr = 0.0001
for step in range(1, 201):
    guesses = predict(sugar)
    errors = guesses - titer
    dw = 2 * np.mean(errors * sugar)   # direction of less wrong for w
    db = 2 * np.mean(errors)          # direction of less wrong for b
    w = w - lr * dw
    b = b - lr * db
    if step % 50 == 0:
        print("step", step, " loss =", round(loss(), 3),
              " w =", round(w, 3), " b =", round(b, 3))

print("learned: w =", round(w, 3), " b =", round(b, 3))
