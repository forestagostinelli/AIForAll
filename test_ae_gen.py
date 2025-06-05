from typing import Optional
import torch
import torch.nn as nn
import numpy as np
from torch.optim.optimizer import Optimizer
import torch.optim as optim
from torch import Tensor
from train_mnist_ae import get_decoder, get_encoder, get_encoder_variational, get_ae

import time
from argparse import ArgumentParser
import pickle
import matplotlib.pyplot as plt


def evaluate_nnet(nnet: nn.Module, data_input_np, fig, axs):
    nnet.eval()
    criterion = nn.MSELoss()

    val_input = torch.tensor(data_input_np).float()
    nnet_output: Tensor = nnet(val_input)

    loss = criterion(nnet_output, val_input)

    plt_idxs = np.linspace(0, val_input.shape[0] - 1, axs.shape[1]).astype(int)
    for plot_num in range(axs.shape[1]):
        ax_in, ax_out = axs[:, plot_num]
        for ax in [ax_in, ax_out]:
            ax.cla()

        plot_idx: int = plt_idxs[plot_num]
        ax_in.imshow(val_input[plot_idx, :].reshape((28, 28)), cmap="gray")
        ax_out.imshow(nnet_output.cpu().data.numpy()[plot_idx, :].reshape((28, 28)), cmap="gray")

    fig.canvas.draw()
    fig.canvas.flush_events()
    fig.show()
    plt.pause(0.01)

    return loss.item()


def print_event(event):
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))


class MoveGraphLine:
    def __init__(self, ax_click, ax_show, decoder: nn.Module, cvae: Optional[int]):
        self.ax_click = ax_click
        self.ax_show = ax_show
        self.decoder: nn.Module = decoder
        self.moved = None
        self.point = None
        self.point_plot = None
        self.pressed = False
        self.start = False
        self.cvae: Optional[int] = cvae

    def mouse_release(self, _):
        if self.pressed:
            self.pressed = False

    def mouse_press(self, event):
        if not (event.inaxes == self.ax_click):
            return

        if self.start:
            return
        self.pressed = True
        self._update_plot(event)

    def mouse_move(self, event):
        if not (event.inaxes == self.ax_click):
            return

        if not self.pressed:
            return

        self._update_plot(event)

    def _update_plot(self, event):
        enc_np = np.array((event.xdata, event.ydata))
        enc_np = np.expand_dims(enc_np, 0)

        if self.point_plot is not None:
            self.point_plot.remove()
        self.point_plot = self.ax_click.scatter(enc_np[0, 0], enc_np[0, 1], marker='*', color='k')

        dec_input = torch.tensor(enc_np).float()
        if self.cvae is not None:
            val_labels = torch.tensor(self.cvae * np.ones((dec_input.shape[0], 1))).float()
            dec_input = torch.cat((dec_input, val_labels), dim=1)

        dec_output = self.decoder(dec_input).cpu().data.numpy()

        self.ax_show.cla()
        self.ax_show.imshow(dec_output[0, :].reshape((28, 28)), cmap="gray")


def train_nnet(nnet: nn.Module, train_input_np: np.ndarray, val_input_np: np.ndarray, fig, axs) -> nn.Module:
    # optimization
    train_itr: int = 0
    batch_size: int = 100
    num_itrs: int = 10000

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

        # forward
        nnet_output_b: Tensor = nnet(data_input_b)

        # cost
        loss = criterion(nnet_output_b, data_input_b)

        # backwards
        loss.backward()

        # step
        optimizer.step()

        # display progress
        if train_itr % display_itrs == 0:
            nnet.eval()
            loss_val = evaluate_nnet(nnet, val_input_np, fig, axs)
            nnet.train()

            print("Itr: %i, lr: %.2E, loss: %.5f, loss_val: %.5f, Time: %.2f" % (
                      train_itr, lr_itr, loss.item(), loss_val, time.time() - start_time))

            start_time = time.time()

        train_itr = train_itr + 1

    return nnet


def plot_color_coded(encoded, val_labels_np, ax):
    for label in range(10):
        label_idxs = np.where(val_labels_np == label)
        ax.scatter(encoded[label_idxs, 0], encoded[label_idxs, 1], alpha=0.7, label=f"{label}")


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--nnet', type=str, required=True, help="")
    parser.add_argument('--vae', action='store_true', default=False, help="")
    parser.add_argument('--cvae', type=int, default=None, help="")
    args = parser.parse_args()

    plt.ion()

    # load nnet
    if args.vae or (args.cvae is not None):
        encoder: nn.Module = get_encoder_variational(args.cvae is not None)
    else:
        encoder: nn.Module = get_encoder()
    encoder.load_state_dict(torch.load(f"{args.nnet}/encoder.pt", weights_only=True))
    decoder: nn.Module = get_decoder(args.cvae is not None)
    decoder.load_state_dict(torch.load(f"{args.nnet}/decoder.pt", weights_only=True))

    nnet: nn.Module = get_ae(encoder, decoder, args.cvae is not None)
    encoder.eval()
    decoder.eval()
    nnet.eval()

    # parse data
    val_input_np, val_labels_np = pickle.load(open("data/mnist/mnist_val.pkl", "rb"))
    if args.cvae is not None:
        cvae_idxs = np.where(val_labels_np == args.cvae)[0]
        val_input_np = val_input_np[cvae_idxs]
        val_labels_np = val_labels_np[cvae_idxs]
    val_input_np = val_input_np.reshape(-1, 28 * 28)

    val_input = torch.tensor(val_input_np).float()
    if args.cvae is not None:
        val_labels = torch.tensor(args.cvae * np.ones((val_input.shape[0], 1))).float()
        ae_input = torch.cat((val_input, val_labels), dim=1)
    else:
        ae_input = val_input

    encoded = encoder(ae_input).cpu().data.numpy()

    fig, axs = plt.subplots(1, 2)
    fig.show()

    # axs[0].scatter(encoded[:, 0], encoded[:, 1], alpha=0.1)
    plot_color_coded(encoded, val_labels_np, axs[0])

    for ax in axs:
        # ax.set_xlim(-5, 5)
        # ax.set_ylim(-5, 5)
        ax.set(adjustable='box', aspect='equal')
    axs[0].legend(bbox_to_anchor=(1.75, -0.2), ncol=5)

    # plt.connect('button_press_event', onclick)
    # _ = fig.canvas.mpl_connect('button_press_event', onclick)
    mgl = MoveGraphLine(axs[0], axs[1], decoder, args.cvae)
    fig.canvas.mpl_connect('button_press_event', mgl.mouse_press)
    fig.canvas.mpl_connect('button_release_event', mgl.mouse_release)
    fig.canvas.mpl_connect('motion_notify_event', mgl.mouse_move)
    fig.show()

    plt.show(block=True)

    """
    while True:
        in_val: str = input("x,y: ")
        if len(in_val) == 0:
            break
        x_val, y_val = in_val.split(",")
        enc_np = np.array((float(x_val), float(y_val)))
        enc_np = np.expand_dims(enc_np, 0)

        axs[0].cla()
        plot_color_coded(encoded, val_labels_np, axs[0])
        axs[0].legend(bbox_to_anchor=(1.75, -0.2), ncol=5)
        axs[0].scatter(enc_np[0, 0], enc_np[0, 1], marker='*', color='k')

        dec_input = torch.tensor(enc_np).float()
        dec_output = decoder(dec_input).cpu().data.numpy()

        axs[1].cla()
        axs[1].imshow(dec_output[0, :].reshape((28, 28)), cmap="gray")
    """


if __name__ == "__main__":
    main()
