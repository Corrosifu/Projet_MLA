# Reimplementing important results from "[Goodfellow, Ian &amp; Shlens, Jonathon &amp; Szegedy, Christian. (2014). Explaining and Harnessing Adversarial Examples. arXiv 1412.6572.](https://arxiv.org/abs/1412.6572) "

## Quick definition by the authors of the article :

> Adversarial examples : inputs formed by applying small but intentionally worst-case perturbations to examples from the dataset, such that the perturbed input results in the model outputting an incorrect answer with high confidence

## Goals :

We tried to reimplement some important empirical results from "[Explaining and harnessing adversarial examples](https://arxiv.org/abs/1412.6572)". We successfully implemented an adversarial example generator as described in the article and we tried it on a shallow softmax network and a maxout network (both linear neural networks). We did not manage to implement GoogLeNet and to try our adversarial examples generator on Tiny ImageNet. We then proved that a Radial Basis Function (RBF) network is more resistant to adversarial attacks than the linear models cited above.

## Framework used :

- Python
- TensorFlow/Keras

## How to run the project :

We mostly used Jupyter Notebooks so you will only need to run the notebooks to use them. You may need to install common ML librairies to run the notebooks :

- tensorflow
- tensorflow_addons

And some more common :

- numpy
- matplotlib.pyplot
- google.colab
- PIL
- ...

## File description :

- fgsm.ipynb is the notebook you will want to run if you want to see our adversarial example generator in action with linear neural networks
- fgsm.py is only the adversarial example generator if you want to import it
- GoogLeNet_Colab is a Google Colab version of our try at implementing GoogLeNet and evaluating adversarial Tiny ImageNet examples with it.
- GoogLeNet_GPU_Server is the version of our try at implementing GoogLeNet and evaluating adversarial Tiny ImageNet examples with it we used on a GPU server.
- RBF_network.ipynb if the notebook you should run if you want to see that non-linear networks are more resistant than linear ones to adversarial attacks
- logistic_regression.ipynb is used to classify 3 and 7s from the MNIST dataset using logistic regression. Then it plots the weights obtained et creates some adversarial example to show the basic working principles of the Fast Gradient Sign Method
- adversarial_training contains code showing that it is possible to adversarially train a deep network
- the rest of the files are saves of the RBF model
