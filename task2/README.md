# Task 2

*EXPERIMENT_LOG.ipynb* -> Includes the thought process behind the design and the step by step building process, along with relevant snippets.

*train.py* -> Script for training the neural network

*numbers_and_letters.py* -> Holds implementation of dataset and the neural network that has been used in training on task2.1 i.e. numbers and letters.

## Running

*Task 2.1*
``` sh
python train.py --dataset numbers_and_letters
```

*Task 2.2*

To train on numbers_only, set the `numbers_only` flag to `True`.

``` sh
python train.py --dataset numbers_and_letters 
```

To use this pretrained model on mnist: (make sure LOAD_MODEL_NAME is set the current model weights to load)

``` sh
python train.py --dataset mnist --pretrained True
```

And to train on MNIST from scratch:

``` sh
python train.py --dataset mnist
```

*Task 2.3*
