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

### 2-dimensional Neural Network classification
`python 2D_train.py --dataset <dataset_name> --steps <training_steps> --lr <learning_rate>`\
Dataset names: lin, xor, half_moons\
Change model in "EDIT HERE" and see how it performs

### Train Neural Network for MNIST Classification
`python mnist_train.py`\
Change model in "EDIT HERE" and see how it performs

### Test Neural Network on MNIST Classification
`python mnist_test.py --nnet <nnet_model> --dim <size_of_plot>`\
**Switches**:\
Linear model: `--lin`

**Examples:**\
Linear model: `python mnist_test.py --nnet models/mnist/mnist_lin.pt --dim 500 --lin`\
Neural network: `python mnist_test.py --nnet models/mnist/mnist_lenet.pt --dim 500`

### MNIST Autoencoders

#### Training
Autoencoder: `python train_mnist_ae.py --save_dir models/mnist_ae/`\
Variational autoencoder: `python train_mnist_ae.py --save_dir models/mnist_vae/ --vae`\
Conditional variational autoencoder: `python train_mnist_ae.py --save_dir models/mnist_cvae/ --cvae`

#### Viewing latent space and reconstructions
Autoencoder: `python test_ae_gen.py --nnet models/mnist_ae/`\
Variational autoencoder: `python test_ae_gen.py --nnet models/mnist_vae/ --vae`\
Conditional variational autoencoder: `python test_ae_gen.py --nnet models/mnist_cvae/ --cvae <number>`

## Recordings
* 06/02: Linear Models | [recording](https://nam02.safelinks.protection.outlook.com/?url=https%3A%2F%2Fsc-edu.zoom.us%2Frec%2Fshare%2FjMkw9j4lUtQUJ54GnVyx06k6kGfeXH7AvU5HNkSB6iD3KsxPQlRWW9dBbNqNXZTn.5XMjyRgPixHA51OE%3FstartTime%3D1748872185000&data=05%7C02%7CFORESTA%40cse.sc.edu%7Ca22c69c849004b24c0e508dda1f7c79a%7C4b2a4b19d135420e8bb2b1cd238998cc%7C0%7C0%7C638844807751730193%7CUnknown%7CTWFpbGZsb3d8eyJFbXB0eU1hcGkiOnRydWUsIlYiOiIwLjAuMDAwMCIsIlAiOiJXaW4zMiIsIkFOIjoiTWFpbCIsIldUIjoyfQ%3D%3D%7C0%7C%7C%7C&sdata=usi4%2FgG%2FNgfvrDl2bDv24qfEWQlAPapZdZE%2BZel5i1Q%3D&reserved=0)
* 06/03: Neural Networks 1 | [recording](https://sc-edu.zoom.us/rec/play/cjx1H2XgCwQISgU47QF6Avn0XXs7nSa9K5HFPc5VgOUcWYmsRsWZvDrrOMB7YVMxwpagFouvna7f-wR_.GT8zhiyTq_C_FJbG?eagerLoadZvaPages=sidemenu.billing.plan_management&accessLevel=meeting&canPlayFromShare=true&from=share_recording_detail&startTime=1748958853000&componentName=rec-play&originRequestUrl=https%3A%2F%2Fsc-edu.zoom.us%2Frec%2Fshare%2FPd1X8Sg6dewa7U56uUhvl_qJGCLpk3D5UG3VyNRK3BZMe0kPkTzBqo_znKuFTMDj.Ty1UahXEXVl0JJTr%3FstartTime%3D1748958853000)
* 06/04: Neural Networks 2 | [recording](https://nam02.safelinks.protection.outlook.com/?url=https%3A%2F%2Fsc-edu.zoom.us%2Frec%2Fshare%2FfZ_gxEQmwFzi8MLigQ_w021ciBbi7XO_i2t7CFxsxu1XJS1NRgdiXWmf3caD-doB.JcnQxPO50dSpduB7%3FstartTime%3D1749045698000&data=05%7C02%7CFORESTA%40cse.sc.edu%7C569b552996214ef1339008dda44982b8%7C4b2a4b19d135420e8bb2b1cd238998cc%7C0%7C0%7C638847357808262767%7CUnknown%7CTWFpbGZsb3d8eyJFbXB0eU1hcGkiOnRydWUsIlYiOiIwLjAuMDAwMCIsIlAiOiJXaW4zMiIsIkFOIjoiTWFpbCIsIldUIjoyfQ%3D%3D%7C0%7C%7C%7C&sdata=a6v27HpmgIYUyb9gI4nl4b2W1%2FDYWiVmPo1JdJKBkKg%3D&reserved=0)
