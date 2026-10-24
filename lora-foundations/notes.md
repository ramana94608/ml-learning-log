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

## Day 2 (2026-10-21): train vs validation, overfitting

- Overfitting is when the model memorizes the practice readings including their noise, so it fails on new readings because the new noise is different.
- Two thermometers: train loss = score on the practice readings, val loss = score on readings the model never saw. Practice great, exam terrible = memorized, not learned.
- Run result: honest line (degree 1) train 1.073, val 1.245. Memorizer (degree 16) train 0.777, val 9.607.

## Day 3 (2026-10-22): the adapter, and alpha

- LoRa learns patterns from examples, doesn’t take orders
- The base model stays frozen. The adapter is a small trainable side path: rank 8 means two skinny sheets, 16,000 numbers against a frozen million, about 1.6%.
- Alpha is the volume knob, and the real scale is alpha divided by rank. Alpha equal to rank means volume 1. Alpha twice the rank means the adapter hits with double impact and drowns the base model (the Kaggle board case: looping, score 0.0, while its loss still fell to 0.51).

## Day 4 (2026-10-23): the volume knob, in numbers

- The knob changes how hard the notes push, not where they push.
- Multiplying two lists: line them up, multiply each pair, add the results. A @ x = 1*1 + 1*2 = 3. A matrix is recipes stacked; each row runs the same pair trick on the input.
- Script: day4_volume_knob.py. Frozen W passes the reading [1, 2] through untouched, the nudge is [3, 0]. Run: s = 0 gives [1, 2], s = 0.5 gives [2.5, 2], s = 1 gives [4, 2], s = 2 gives [7, 2]: at double volume the notes drag the scientist's 1 to 7.
