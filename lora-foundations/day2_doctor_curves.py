"""
Day 2: reading training like a doctor.
You run this in the VS Code terminal: python3, then paste each block.
Same story as Day 1 (sugar in -> titer out), but now 60 noisy readings.
"""
import numpy as np

np.random.seed(7)
sugar = np.linspace(10., 50., 60)
# True relationship: titer = 0.5 * sugar, plus measurement noise.
titer = 0.5 * sugar + np.random.normal(0, 1.0, 60)

# REP 1: two thermometers.
# Shuffle the readings, then split: the model trains on 45, and the last 15
# are the validation set, readings the model never trains on, like a blind
# control. Shuffling keeps both sets in the same range, so the only
# difference is "trained on" vs "never seen".
order = np.random.permutation(60)
train_sugar, train_titer = sugar[order[:45]], titer[order[:45]]
val_sugar, val_titer = sugar[order[45:]], titer[order[45:]]

w = 0.0
b = 0.0

def predict(x):
    return w * x + b

def loss(xs, ys):
    # Loss = average wrongness, squared, on a given set of readings.
    guesses = predict(xs)
    errors = guesses - ys
    return float(np.mean(errors ** 2))

lr = 0.0001
for step in range(1, 401):
    guesses = predict(train_sugar)
    errors = guesses - train_titer
    dw = 2 * np.mean(errors * train_sugar)
    db = 2 * np.mean(errors)
    w = w - lr * dw
    b = b - lr * db
    if step % 100 == 0:
        print("step", step,
              " train_loss =", round(loss(train_sugar, train_titer), 3),
              " val_loss   =", round(loss(val_sugar, val_titer), 3))

print("learned: w =", round(w, 3), " b =", round(b, 3))

# REP 2: the memorizer, on purpose.
# A degree-16 curve has far more knobs than it needs. Watch the two
# thermometers disagree.
coef12 = np.polyfit(train_sugar, train_titer, 16)
coef1 = np.polyfit(train_sugar, train_titer, 1)

def poly_loss(coef, xs, ys):
    guesses = np.polyval(coef, xs)
    return float(np.mean((guesses - ys) ** 2))

print()
print("honest line (degree 1): train_loss =",
      round(poly_loss(coef1, train_sugar, train_titer), 3),
      " val_loss =", round(poly_loss(coef1, val_sugar, val_titer), 3))
print("memorizer   (degree 16): train_loss =",
      round(poly_loss(coef12, train_sugar, train_titer), 3),
      " val_loss =", round(poly_loss(coef12, val_sugar, val_titer), 3))

# The doctor's chart: training points, the honest line, the wiggly memorizer.
import matplotlib.pyplot as plt

xs = np.linspace(10., 50., 200)
plt.scatter(train_sugar, train_titer, s=12, label="training readings")
plt.scatter(val_sugar, val_titer, s=12, marker="x", label="validation readings")
plt.plot(xs, np.polyval(coef1, xs), label="degree 1 (honest)")
plt.plot(xs, np.polyval(coef12, xs), label="degree 16 (memorizer)")
plt.xlabel("sugar (g/L)")
plt.ylabel("titer (g/L)")
plt.legend()
plt.show()
