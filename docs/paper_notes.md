# Notes on the paper

> Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy. **Explaining and Harnessing Adversarial
> Examples.** ICLR 2015. [arXiv:1412.6572](https://arxiv.org/abs/1412.6572) (v3, 20 March 2015).

Section numbers below refer to that version. All MNIST experiments use pixels in [0, 1] and
`eps = 0.25` unless stated otherwise.

## The claim

Adversarial examples are not a quirk of highly non-linear, over-fitted networks. They are a
consequence of **linear behaviour in high-dimensional spaces**: models that are easy to optimise
(ReLU / maxout networks, LSTMs, softmax regression) are easy to perturb.

## 1. The linear explanation (section 3)

For a weight vector `w` with `n` dimensions and an input perturbation `eta` with `||eta||_inf < eps`:

```
w^T (x + eta) = w^T x + w^T eta        maximised by  eta = eps * sign(w)
w^T eta = eps * ||w||_1 = eps * m * n          (m = mean magnitude of a weight)
```

`||eta||_inf` does not grow with `n`, but the activation change grows **linearly** with it: many
imperceptible changes add up to one large change of the output. Implemented as
`fgsm.activation_shift`; checked in `tests/test_fgsm.py::test_linear_activation_shift_grows_with_the_dimension`.

## 2. The fast gradient sign method (section 4)

Linearise the cost around the current parameters:

```
eta = eps * sign( grad_x J(theta, x, y) )        x_adv = x + eta
```

One back-propagation, no inner optimisation (unlike box-constrained L-BFGS). Reported on MNIST:
shallow softmax **99.9 %** error with **79.3 %** average confidence; maxout network **89.4 %** error,
**97.6 %** confidence. Code: `fgsm.fgsm`.

## 3. Logistic regression is the exact case (section 5, Figure 2)

For `P(y=1|x) = sigmoid(w^T x + b)` the gradient sign is `-y sign(w)` (labels in {-1, 1}), so the
method is not an approximation but the worst case inside the max-norm box. On 3 vs 7: 1.6 % clean
error, **99 %** error on the adversarial examples. Code: `fgsm.fgsm_logistic_exact`, verified
against autodiff in `tests/test_fgsm.py`.

## 4. Adversarial training (section 6)

```
J~(theta, x, y) = alpha * J(theta, x, y) + (1 - alpha) * J(theta, x + eps * sign(grad_x J), y)
```

with `alpha = 0.5`, adversarial examples regenerated from the *current* model at every step.
Maxout + dropout: test error 0.94 % -> **0.84 %**. With 1600 units per layer and early stopping on
the **adversarial** validation error: 0.782 % (best permutation-invariant MNIST result at the time).
The adversarial error falls from **89.4 % to 17.9 %**, but the confidence on remaining mistakes is
still **81.4 %**. Code: `training.fit_adversarial`.

Controls: training with random `+-eps` noise or `U(-eps, eps)` noise does *not* help (86.2 % and
90.4 % error on FGSM examples). Linear models lack the capacity to resist (the universal
approximation theorem needs a hidden layer).

## 5. RBF networks (section 7)

A shallow RBF network with no hidden layer, `p(y = 1 | x) = exp(-(x - mu)^T beta (x - mu))`,
answers confidently only near its centres. It gets **55.4 %** error on FGSM examples but its
confidence on those mistakes is only **1.2 %** (60.6 % average confidence on clean data):
"naturally immune" in the sense of knowing it does not know. Code: `models.RBFOutput`.

## 6. Transferability and other hypotheses (sections 8-9)

Adversarial examples generalise across models because the *direction* of the perturbation matters,
not a precise location (Figure 4: adversarial regions are contiguous in `eps`). Maxout-generated
examples are classified like maxout by softmax regression 54.6 % of the time (84.6 % when both
are wrong) and by the RBF network 16.0 % (54.3 %). Generative training (MP-DBM: 97.5 % error) and
ensembles (91.1 %) do not help.

## 7. Rubbish class examples (appendix A)

Feeding `N(0, I)` noise: maxout 98.35 % error (92.8 % confidence), softmax regression 59.8 %
(70.8 %), RBF 0 %.

## Not covered here

Figure 1 (GoogLeNet on ImageNet), the CIFAR-10 convolutional maxout result (87.15 %), the MP-DBM and
the 12-model ensemble. The original project tried GoogLeNet on Tiny ImageNet but never trained
it (see `legacy/README.md`).
