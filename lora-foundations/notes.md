# LoRA foundations: my notes, my words

## Day 1 (2026-10-20): what training is

- training is minimizing the loss of error from the prediction
- Loss: square each error first, then average. Square-then-average, in that
  order, so misses cannot cancel each other out.
- dw is the average error weighted by the input; db is the plain average
  error. The update walks the other way: w = w - lr * dw.
- Learning rate: step you want / gradient you have. Too big overshoots the
  bottom and the loss explodes (one step at lr 0.01: loss ~120,000).
- Run result: loss 275.4 -> 0.022 by step 50, w = 0.5, b = 0.014.
