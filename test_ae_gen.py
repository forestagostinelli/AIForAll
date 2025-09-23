from typing import Optional
import torch
import torch.nn as nn
import numpy as np
from numpy.typing import NDArray
from train_mnist_ae import get_decoder, get_encoder, get_encoder_variational, get_ae, make_cond
from utils.mnist_utils import colors

from argparse import ArgumentParser
import pickle
import matplotlib.pyplot as plt


def print_event(event):
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))


def make_cond_arr(digit: int, color_idx: int, cmnist: bool, num: int) -> NDArray:
    cond_arr = np.array([digit] * num)
    if cmnist:
        cond_arr_color = np.array([color_idx] * num)
        cond_arr = np.stack((cond_arr, cond_arr_color), axis=1)

    return cond_arr


class MoveGraphLine:
    def __init__(self, ax_click, ax_show, decoder: nn.Module, digit: Optional[int], color_idx: int, cmnist: bool):
        self.ax_click = ax_click
        self.ax_show = ax_show
        self.decoder: nn.Module = decoder
        self.moved = None
        self.point = None
        self.point_plot = None
        self.pressed = False
        self.start = False
        self.digit: Optional[int] = digit
        self.color_idx: int = color_idx
        self.cmnist: bool = cmnist

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
        if self.digit is not None:
            cond_arr: NDArray = make_cond_arr(self.digit, self.color_idx, self.cmnist, 1)
            _, cond_dec = make_cond(self.cmnist, cond_arr, False)
            dec_output = self.decoder(dec_input, cond=torch.tensor(cond_dec)).cpu().data.numpy()
        else:
            dec_output = self.decoder(dec_input).cpu().data.numpy()

        self.ax_show.cla()
        if dec_output.shape[3] == 1:
            self.ax_show.imshow(dec_output[0, :], cmap="gray")
        else:
            self.ax_show.imshow(dec_output[0, :])


def plot_color_coded(encoded, val_labels_np, ax):
    for label in range(int(np.max(val_labels_np)) + 1):
        label_idxs = np.where(val_labels_np == label)
        ax.scatter(encoded[label_idxs, 0], encoded[label_idxs, 1], alpha=0.7, label=f"{label}")


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--nnet', type=str, required=True, help="")
    parser.add_argument('--vae', action='store_true', default=False, help="")
    parser.add_argument('--cmnist', action='store_true', default=False, help="")
    parser.add_argument('--digit', type=int, default=None, help="-1 is no digit specified")
    parser.add_argument('--color', type=str, default=None, help="red, green, blue, yellow, purple. "
                                                                "none is no color specified.")
    args = parser.parse_args()

    plt.ion()

    cvae: bool = args.digit is not None
    # load nnet
    if args.vae or cvae:
        encoder: nn.Module = get_encoder_variational(cvae, args.cmnist)
    else:
        encoder: nn.Module = get_encoder(args.cmnist)
    encoder.load_state_dict(torch.load(f"{args.nnet}/encoder.pt", weights_only=True))
    decoder: nn.Module = get_decoder(cvae, args.cmnist)
    decoder.load_state_dict(torch.load(f"{args.nnet}/decoder.pt", weights_only=True))

    nnet: nn.Module = get_ae(encoder, decoder, cvae)
    encoder.eval()
    decoder.eval()
    nnet.eval()

    # parse data
    color_idx: int = -1
    if args.color is not None:
        if args.color == "none":
            color_idx = len(colors)
        else:
            color_idx: int = [x[0] for x in colors].index(args.color)
    if args.cmnist:
        val_input_np, val_labels_np = pickle.load(open("data/mnist/mnist_val_color.pkl", "rb"))
    else:
        val_input_np, val_labels_np = pickle.load(open("data/mnist/mnist_val.pkl", "rb"))
        val_input_np = np.expand_dims(val_input_np, 3)
    if cvae:
        if args.digit == -1:
            cvae_idxs = np.arange(0, val_labels_np.shape[0])
        else:
            if args.cmnist:
                cvae_idxs = np.where(val_labels_np[:, 0] == args.digit)[0]
            else:
                cvae_idxs = np.where(val_labels_np == args.digit)[0]
        if args.cmnist:
            if color_idx == len(colors):
                cvae_idxs_color = np.arange(0, val_labels_np.shape[0])
            else:
                cvae_idxs_color = np.where(val_labels_np[:, 1] == color_idx)[0]
            cvae_idxs = np.intersect1d(cvae_idxs_color, cvae_idxs)
        val_input_np = val_input_np[cvae_idxs]
        val_labels_np = val_labels_np[cvae_idxs]

    val_input = torch.tensor(val_input_np).float()
    if cvae:
        cond_arr: NDArray = make_cond_arr(args.digit, color_idx, args.cmnist, val_input.size()[0])
        cond_enc, _ = make_cond(args.cmnist, cond_arr, False)
        encoded = encoder(val_input, cond=torch.tensor(cond_enc)).cpu().data.numpy()
    else:
        encoded = encoder(val_input).cpu().data.numpy()

    fig, axs = plt.subplots(1, 2)
    fig.show()

    # axs[0].scatter(encoded[:, 0], encoded[:, 1], alpha=0.1)
    if args.cmnist:
        plot_color_coded(encoded, val_labels_np[:, 0], axs[0])
    else:
        plot_color_coded(encoded, val_labels_np, axs[0])

    for ax in axs:
        # ax.set_xlim(-5, 5)
        # ax.set_ylim(-5, 5)
        ax.set(adjustable='box', aspect='equal')
    axs[0].legend(bbox_to_anchor=(1.75, -0.2), ncol=5)

    # plt.connect('button_press_event', onclick)
    # _ = fig.canvas.mpl_connect('button_press_event', onclick)
    mgl = MoveGraphLine(axs[0], axs[1], decoder, args.digit, color_idx, args.cmnist)
    fig.canvas.mpl_connect('button_press_event', mgl.mouse_press)
    fig.canvas.mpl_connect('button_release_event', mgl.mouse_release)
    fig.canvas.mpl_connect('motion_notify_event', mgl.mouse_move)
    fig.show()

    plt.show(block=True)


if __name__ == "__main__":
    main()
