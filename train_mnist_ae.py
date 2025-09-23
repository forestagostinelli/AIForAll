from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from numpy.typing import NDArray
from torch.optim.optimizer import Optimizer
import torch.optim as optim
from torch import Tensor
from utils.mnist_utils import colors
import time
from argparse import ArgumentParser
import pickle
import os
import matplotlib.pyplot as plt


def evaluate_nnet(nnet: nn.Module, val_input_np, val_labels_np, fig, axs, cvae: bool, cmnist: bool):
    nnet.eval()
    criterion = nn.MSELoss()

    val_input = torch.tensor(val_input_np).float()

    if cvae:
        cond_enc, cond_dec = make_cond(cmnist, val_labels_np, False)
        nnet_output: Tensor = nnet(val_input, torch.tensor(cond_enc), torch.tensor(cond_dec))
    else:
        nnet_output: Tensor = nnet(val_input)

    loss = criterion(nnet_output, val_input)

    plt_idxs = np.linspace(0, val_input.shape[0] - 1, axs.shape[1]).astype(int)
    for plot_num in range(axs.shape[1]):
        ax_in, ax_out = axs[:, plot_num]
        for ax in [ax_in, ax_out]:
            ax.cla()

        plot_idx: int = int(plt_idxs[plot_num])
        ax_in.imshow(val_input[plot_idx, :], cmap="gray")
        ax_out.imshow(nnet_output.cpu().data.numpy()[plot_idx, :], cmap="gray")

    fig.canvas.draw()
    fig.canvas.flush_events()
    fig.show()
    plt.pause(0.01)

    return loss.item()


def get_act_fn(act: str):
    act = act.upper()
    if act == "RELU":
        act_fn = nn.ReLU()
    elif act == "SIGMOID":
        act_fn = nn.Sigmoid()
    elif act == "TANH":
        act_fn = nn.Tanh()
    else:
        raise ValueError("Un-defined activation type %s" % act)

    return act_fn


class FullyConnectedModel(nn.Module):
    def __init__(self, input_dim: int, dims: List[int], acts: List[str]):
        super().__init__()
        self.layers: nn.ModuleList[nn.ModuleList] = nn.ModuleList()

        # layers
        for dim, act in zip(dims, acts):
            module_list = nn.ModuleList()

            # linear
            linear_layer = nn.Linear(input_dim, dim)
            module_list.append(linear_layer)

            # activation
            if act.upper() != "LINEAR":
                module_list.append(get_act_fn(act))

            self.layers.append(module_list)

            input_dim = dim

    def forward(self, x):
        x = x.float()

        module_list: nn.ModuleList
        for module_list in self.layers:
            for module in module_list:
                x = module(x)

        return x


class EncoderConv(nn.Module):
    def __init__(self, num_channels: int, latent_dim=2):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, 3, stride=2, padding=1)   # 28x28 -> 14x14
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)  # 14x14 -> 7x7
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1) # 7x7 -> 4x4
        self.bn3 = nn.BatchNorm2d(32)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 4 * 4, latent_dim)

    def forward(self, x: Tensor, cond: Optional[Tensor] = None):
        x = x.permute([0, 3, 1, 2])
        if cond is not None:
            x = torch.concat((x, cond.float()), dim=1)
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = nn.functional.relu(self.bn3(self.conv3(x)))
        x = self.flatten(x)
        z = self.fc(x)  # bottleneck
        return z


class DecoderConv(nn.Module):
    def __init__(self, num_channels: int, latent_dim=2):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 32 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=0)  # 4x4 -> 7x7
        self.bn1 = nn.BatchNorm2d(32)
        self.deconv2 = nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1)   # 7x7 -> 14x14
        self.bn2 = nn.BatchNorm2d(32)
        self.deconv3 = nn.ConvTranspose2d(32, num_channels, 3, stride=2, padding=1, output_padding=1)    # 14x14 -> 28x28

    def forward(self, z, cond: Optional[Tensor] = None):
        if cond is not None:
            z = torch.cat((z, cond.float()), dim=1)
        x = self.fc(z)
        x = x.view(-1, 32, 4, 4)  # reshape into conv feature map
        x = nn.functional.relu(self.bn1(self.deconv1(x)))
        x = nn.functional.relu(self.bn2(self.deconv2(x)))
        x = nn.functional.sigmoid(self.deconv3(x))
        x = x.permute([0, 2, 3, 1])
        return x


class VAE(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder: nn.Module = encoder
        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x, cond: Optional[Tensor] = None):
        x = self.encoder(x, cond=cond)
        mu = x[:, :2]
        logvar = x[:, 2:]
        sigma = torch.exp(logvar / 2.0)
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)

        return z

def get_encoder(cmnist: bool) -> nn.Module:
    if cmnist:
        num_channels: int = 3
    else:
        num_channels: int = 1

    return EncoderConv(num_channels)


def get_encoder_variational(cvae: bool, cmnist: bool) -> nn.Module:
    if cmnist:
        if cvae:
            num_channels: int = 3 + 11 + (len(colors) + 1)
        else:
            num_channels: int = 3
    else:
        if cvae:
            num_channels: int = 1 + 11
        else:
            num_channels: int = 1

    return VAE(EncoderConv(num_channels, latent_dim=2 * 2))


def get_decoder(cvae: bool, cmnist: bool) -> nn.Module:
    latent_dim: int = 2
    if cmnist:
        num_channels: int = 3
        if cvae:
            latent_dim = 2 + 11 + (len(colors) + 1)

    else:
        num_channels: int = 1
        if cvae:
            latent_dim = 2 + 11

    return DecoderConv(num_channels, latent_dim=latent_dim)


class Autoencoder(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


class AutoencoderCond(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, cond_enc: Tensor, cond_dec):
        x = self.encoder(x, cond=cond_enc)
        x = self.decoder(x, cond=cond_dec)

        return x


def get_ae(encoder: nn.Module, decoder: nn.Module, cvae: bool) -> nn.Module:
    if cvae:
        return AutoencoderCond(encoder, decoder)
    else:
        return Autoencoder(encoder, decoder)


def one_hot_images(indices, max_int: int) -> np.ndarray:
    """
    Create one-hot KxNx28x28 tensor.

    Args:
        indices: Array of length N with values in [0, max_int-1]
        max_int: K, number of categories

    Returns:
        np.ndarray: Shape (N, K, 28, 28), dtype=np.float32
    """
    # initialize zeros
    out = np.zeros((indices.shape[0], max_int, 28, 28), dtype=np.float32)

    # fill in one-hot slices
    for k, idx in enumerate(indices):
        out[k, idx] = 1.0  # whole 28x28 slice is ones

    return out


def make_cond(cmnist: bool, labels: NDArray, rand_dontcare: bool) -> Tuple[NDArray, NDArray]:
    digit_num_cat: int = 11
    colors_num_cat: int = len(colors) + 1
    num: int = labels.shape[0]
    if cmnist:
        digit_labels: NDArray = labels[:, 0].copy().astype(int)
    else:
        digit_labels: NDArray = labels.copy().astype(int)

    if rand_dontcare:
        digit_mask = np.random.rand(num) < 0.1
        digit_labels[digit_mask] = 10
    digit_labels_enc = one_hot_images(digit_labels.astype(int), digit_num_cat)
    digit_labels_dec = np.eye(digit_num_cat)[digit_labels]
    if cmnist:
        color_labels: NDArray = labels[:, 1].copy().astype(int)
        if rand_dontcare:
            color_mask = np.random.rand(num) < 0.1
            color_labels[color_mask] = len(colors)
        color_labels_enc = one_hot_images(color_labels.astype(int), colors_num_cat)
        color_labels_dec = np.eye(colors_num_cat)[color_labels]
        cond_enc = np.concatenate((digit_labels_enc, color_labels_enc), axis=1)
        cond_dec = np.concatenate((digit_labels_dec, color_labels_dec), axis=1)
    else:
        cond_enc = digit_labels_enc
        cond_dec = digit_labels_dec

    return cond_enc, cond_dec


def train_nnet(nnet: nn.Module, train_input_np: np.ndarray, train_labels_np: np.ndarray, val_input_np: np.ndarray,
               val_labels_np: np.ndarray, fig, axs, vae: bool, cvae: bool, cmnist: bool) -> nn.Module:
    # optimization
    train_itr: int = 0
    batch_size: int = 200
    num_itrs: int = 10000
    if vae:
        kl_weight: float = 0.005
    else:
        kl_weight: float = 0.01
    if cmnist:
        kl_weight = kl_weight / 10.0

    display_itrs = 100
    criterion = nn.MSELoss()
    lr: float = 0.001
    lr_d: float = 0.99996
    optimizer: Optimizer = optim.Adam(nnet.parameters(), lr=lr)
    # optimizer: Optimizer = optim.SGD(nnet.parameters(), lr=lr, momentum=0.9)

    # initialize status tracking
    start_time = time.time()

    nnet.train()
    max_itrs: int = train_itr + num_itrs

    while train_itr < max_itrs:
        # zero the parameter gradients
        optimizer.zero_grad()
        lr_itr: float = lr * (lr_d ** train_itr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_itr

        # get data
        batch_idxs = np.random.randint(0, train_input_np.shape[0], size=batch_size)
        data_input_b = torch.tensor(train_input_np[batch_idxs]).float()

        if cvae:
            cond_enc, cond_dec = make_cond(cmnist, train_labels_np[batch_idxs], True)
            nnet_output_b: Tensor = nnet(data_input_b, torch.tensor(cond_enc), torch.tensor(cond_dec))
        else:
            nnet_output_b: Tensor = nnet(data_input_b)
        # cost
        loss_recon = criterion(nnet_output_b, data_input_b)
        loss_kl = loss_recon * 0
        if vae or cvae:
            loss_kl = nnet.encoder.kl
        loss = loss_recon + kl_weight * loss_kl

        # backwards
        loss.backward()

        # step
        optimizer.step()

        # display progress
        if train_itr % display_itrs == 0:
            nnet.eval()

            loss_val = evaluate_nnet(nnet, val_input_np, val_labels_np, fig, axs, cvae, cmnist)

            nnet.train()

            print("Itr: %i, lr: %.2E, loss_recon: %.2E, loss_kl: %.2E, loss_val: %.2E, Time: %.2f" % (
                train_itr, lr_itr, loss_recon.item(), loss_kl.item(), loss_val, time.time() - start_time))

            start_time = time.time()

        train_itr = train_itr + 1

    return nnet


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--save_dir', type=str, default=None, help="")
    parser.add_argument('--vae', action='store_true', default=False, help="")
    parser.add_argument('--cvae', action='store_true', default=False, help="")
    parser.add_argument('--cmnist', action='store_true', default=False, help="")
    args = parser.parse_args()
    plt.ion()

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    # parse data
    if args.cmnist:
        train_input_np, train_labels_np = pickle.load(open("data/mnist/mnist_train_color.pkl", "rb"))
        val_input_np, val_labels_np = pickle.load(open("data/mnist/mnist_val_color.pkl", "rb"))
    else:
        train_input_np, train_labels_np = pickle.load(open("data/mnist/mnist_train.pkl", "rb"))
        train_input_np = np.expand_dims(train_input_np, 3)

        val_input_np, val_labels_np = pickle.load(open("data/mnist/mnist_val.pkl", "rb"))
        val_input_np = np.expand_dims(val_input_np, 3)

    print(f"Training input shape: {train_input_np.shape}, Validation data shape: {val_input_np.shape}")
    fig, axs = plt.subplots(2, 3)
    fig.show()
    plt.pause(0.01)

    # get nnet
    start_time = time.time()
    if args.vae or args.cvae:
        encoder: nn.Module = get_encoder_variational(args.cvae, args.cmnist)
        decoder: nn.Module = get_decoder(args.cvae, args.cmnist)
    else:
        encoder: nn.Module = get_encoder(args.cmnist)
        decoder: nn.Module = get_decoder(args.cvae, args.cmnist)

    ae: nn.Module = get_ae(encoder, decoder, args.cvae)

    train_nnet(ae, train_input_np, train_labels_np, val_input_np, val_labels_np, fig, axs, args.vae, args.cvae,
               args.cmnist)
    loss = evaluate_nnet(ae, val_input_np, val_labels_np, fig, axs, args.cvae, args.cmnist)
    print(f"Loss: %.5f, Time: %.2f seconds" % (loss, time.time() - start_time))

    torch.save(encoder.state_dict(), f"{args.save_dir}/encoder.pt")
    torch.save(decoder.state_dict(), f"{args.save_dir}/decoder.pt")


if __name__ == "__main__":
    main()
