# Original notebooks (kept for reference)

These are the first version of the project, untouched. They no longer run as is and contain issues
found when reviewing them against the paper; the maintained code is the `adv_repro` package at the
repository root, which reproduces the same experiments with tests.

| File | What it does |
|---|---|
| `fgsm.py`, `fgsm.ipynb` | FGSM generator, shallow softmax and maxout under attack, first adversarial training |
| `logistic_regression.ipynb` | 3 vs 7 logistic regression, weights and sign of the weights (paper, Figure 2) |
| `RBF_network.ipynb` | RBF layer and attack |
| `adversarial_training.ipynb` | adversarial training of a maxout network, compared with a baseline |
| `GoogLeNet_*.ipynb` | attempt to train GoogLeNet on Tiny ImageNet (never completed) |

## Known issues

1. **Images and labels are mismatched in the "train" evaluations.** `x_train` and `y_train` are
   built by two separate passes over a `tf.data` pipeline that calls `shuffle()`, which reshuffles
   at every pass; the adversarial examples come from a third pass. Only about 1 pair in 1000 stays
   aligned. The "accuracy on adversarial examples" on the **training** set (10.1 % in `fgsm.ipynb`)
   is therefore chance level, not a result. The **test** set is not shuffled, so test numbers are valid.
2. **`tensorflow_addons` is end-of-life** (and `requierment.txt` was empty), so the notebooks cannot
   be installed on a current TensorFlow. The new package implements maxout natively.
3. **The "shallow maxout" is not the paper's maxout:** `Dense(100) + Maxout(50) + Dense(10) + Dense(10)`
   versus two maxout layers of 240 units x 5 pieces with dropout. Its 99.8 % error on FGSM examples
   (paper: 89.4 %) is probably partly an architecture difference.
4. **The RBF model is not the paper's either:** a hidden layer of 1000 RBF units followed by
   `Dense(10, sigmoid)` trained with a sparse categorical cross-entropy for **one epoch** (74.9 %
   clean accuracy). Under attack its accuracy drops to 9.8 % (chance) with a mean confidence of 0.097.
   The low confidence agrees with the paper's qualitative claim, but the accuracy does not: the
   old README's statement that RBF networks were *proved* more resistant was too strong.
5. **`adversarial_training.ipynb`:** the baseline ends with `Dense(20)` for a 10-class problem.
6. **GoogLeNet:** `model.fit` fails with a `ValueError` (the notebook loads Tiny ImageNet, then
   overwrites the data with padded MNIST while keeping 200-class targets).
7. Outputs contain absolute local paths of the original machines.
