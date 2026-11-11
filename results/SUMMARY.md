| Metric | Paper | This repo | Diff (pts) | Source |
|---|---|---|---|---|
| Logistic regression 3 vs 7: clean error | 1.6 % | 2.2 % | +0.6 | Fig. 2c |
| Logistic regression 3 vs 7: error on FGSM examples | 99.0 % | 100.0 % | +1.0 | Fig. 2d, eps = .25 |
| Softmax regression: error on FGSM examples | 99.9 % | 100.0 % | +0.1 | Sec. 4 |
| Softmax regression: confidence on those mistakes | 79.3 % | 90.5 % | +11.2 | Sec. 4 |
| Maxout: error on FGSM examples | 89.4 % | 91.9 % | +2.5 | Sec. 4 |
| Maxout: confidence on those mistakes | 97.6 % | 94.5 % | -3.1 | Sec. 4 |
| Maxout + dropout: clean test error | 0.9 % | 1.8 % | +0.8 | Sec. 6 (dropout, no adversarial training) |
| Adversarially trained maxout: clean test error | 0.8 % | 2.1 % | +1.3 | Sec. 6 (240 units); 0.782 % with 1600 units |
| Adversarially trained maxout: error on FGSM examples | 17.9 % | 10.8 % | -7.1 | Sec. 6 (1600 units) |
| Adversarially trained maxout: confidence on mistakes | 81.4 % | 70.0 % | -11.4 | Sec. 6 |
| FGSM examples of original maxout: error on adv. trained | 19.6 % | 23.3 % | +3.7 | Sec. 6 |
| FGSM examples of adv. trained maxout: error on original | 40.9 % | 72.3 % | +31.4 | Sec. 6 |
| Control, random +-eps noise: error on FGSM examples | 86.2 % | 68.4 % | -17.8 | Sec. 6 control (random +-eps noise) |
| Control, random +-eps noise: confidence on mistakes | 97.3 % | 94.2 % | -3.1 | Sec. 6 control |
| Control, uniform noise: error on FGSM examples | 90.4 % | 88.9 % | -1.5 | Sec. 6 control (U(-eps, eps) noise) |
| Control, uniform noise: confidence on mistakes | 97.8 % | 96.9 % | -0.9 | Sec. 6 control |
| Shallow RBF network: error on FGSM examples | 55.4 % | 85.4 % | +30.0 | Sec. 7 |
| Shallow RBF network: confidence on those mistakes | 1.2 % | 12.2 % | +11.0 | Sec. 7 (confidence on mistakes) |
| Shallow RBF network: average confidence on clean data | 60.6 % | 25.4 % | -35.2 | Sec. 7 |
| Gaussian noise inputs, softmax regression: error | 59.8 % | 92.6 % | +32.8 | App. A |
| Gaussian noise inputs, softmax regression: confidence | 70.8 % | 85.9 % | +15.1 | App. A |
| Gaussian noise inputs, maxout: error | 98.4 % | 98.6 % | +0.2 | App. A |
| Gaussian noise inputs, maxout: confidence | 92.8 % | 92.9 % | +0.1 | App. A |
| Gaussian noise inputs, RBF: error | 0.0 % | 0.0 % | +0.0 | App. A |
| Maxout's class predicted by softmax regression | 54.6 % | 56.1 % | +1.5 | Sec. 8 |
| Maxout's class predicted by the RBF network | 16.0 % | 25.0 % | +9.0 | Sec. 8 |
| ... same, among cases where both are wrong (softmax) | 84.6 % | 65.5 % | -19.1 | Sec. 8 |
| ... same, among cases where both are wrong (RBF) | 54.3 % | 58.4 % | +4.1 | Sec. 8 |

**Beyond the paper: PGD (20 steps) versus FGSM, eps = 0.25**

| Model | FGSM error | PGD error |
|---|---|---|
| maxout | 91.9 % | 100.0 % |
| maxout_adv_trained | 10.8 % | 98.2 % |
