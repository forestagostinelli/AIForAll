# AI for All
Code to reproduce instructional material for AI for All.

## Installation
If a conda session is active run:\
`conda deactivate`

Then run:\
`conda env create -f environment.yml`\
`conda activate aiforall`

## Running Scripts

### Linear regression
`python linear_regression.py --lr <learning_rate> --steps <training_steps> --bias <dataset bias>`\
**Switches**:\
No bias parameter: `--no_bias`

**Examples:**\
No bias parameter: `python linear_regression.py --lr 0.1 --no_bias`\
No bias parameter with biased data: `python linear_regression.py --lr 0.1 --bias 3 --no_bias`\
Bias parameter with biased data: `python linear_regression.py --lr 0.1 --bias 3`


### Logistic regression
`python logistic_regression.py --lr <learning_rate> --steps <training_steps>`

**Examples:**\
`python logistic_regression.py`

### Neural Network classification (2-d)
`python nnet_classification.py --dataset <dataset_name> --steps <training_steps> --lr <learning_rate>`\
Dataset names: lin, xor, half_moons

### MNIST Classification
`python mnist_test.py --nnet <nnet_model> --dim <size_of_plot>`\
**Switches**:\
Linear model: `--lin`

**Examples:**\
Linear model: `python mnist_test.py --nnet models/mnist/mnist_lin.pt --dim 500 --lin`\
Neural network: `python mnist_test.py --nnet models/mnist/mnist_lenet.pt --dim 500`

### Train MNIST
`python train_mnist.py`

## Recordings
* 06/02: Linear Models | [recording](https://nam02.safelinks.protection.outlook.com/?url=https%3A%2F%2Fsc-edu.zoom.us%2Frec%2Fshare%2FjMkw9j4lUtQUJ54GnVyx06k6kGfeXH7AvU5HNkSB6iD3KsxPQlRWW9dBbNqNXZTn.5XMjyRgPixHA51OE%3FstartTime%3D1748872185000&data=05%7C02%7CFORESTA%40cse.sc.edu%7Ca22c69c849004b24c0e508dda1f7c79a%7C4b2a4b19d135420e8bb2b1cd238998cc%7C0%7C0%7C638844807751730193%7CUnknown%7CTWFpbGZsb3d8eyJFbXB0eU1hcGkiOnRydWUsIlYiOiIwLjAuMDAwMCIsIlAiOiJXaW4zMiIsIkFOIjoiTWFpbCIsIldUIjoyfQ%3D%3D%7C0%7C%7C%7C&sdata=usi4%2FgG%2FNgfvrDl2bDv24qfEWQlAPapZdZE%2BZel5i1Q%3D&reserved=0)
