from typing import Tuple, Set
import torch
import torch.nn as nn
import numpy as np
from argparse import ArgumentParser
from train_mnist_ae import get_encoder_variational, get_decoder, get_ae

from PIL import Image
import pylab
import tkinter as tk
from tkinter import Canvas


line_id = None


def softmax(data: np.ndarray):
    data_exp = np.exp(data)
    return data_exp/np.sum(data_exp)


def run_nnet(nnet: nn.Module, im_np: np.ndarray) -> np.ndarray:
    im_np = np.expand_dims(im_np, 0)
    im_np = im_np.reshape((1, 28 * 28))
    output = nnet(torch.tensor(im_np).float()).cpu().data.numpy()[0]

    return output


def norm_0_1(data: np.ndarray) -> np.ndarray:
    data = data - np.min(data)
    data = data/np.max(data)

    return data


def get_off_idxs_change(nnet: nn.Module, class_val: int, on_idxs_set: Set[int]) -> Tuple[Set[int], int]:
    example: np.ndarray = np.zeros(28 * 28, dtype=int)
    example[np.array(list(on_idxs_set))] = 1
    class_val_new: int = run_nnet(nnet, example.reshape(28, 28)).argmax()
    if class_val_new != class_val:
        return on_idxs_set, class_val_new
    else:
        for elem in on_idxs_set:
            on_idxs_set_new: Set[int] = on_idxs_set.copy()
            on_idxs_set_new.remove(elem)
            on_idxs_set_ret, class_val_new = get_off_idxs_change(nnet, class_val, on_idxs_set_new)
            if class_val_new != class_val:
                return on_idxs_set_ret, class_val_new


def get_canvas_img(canvas: Canvas, dim: int, box_start: int, box_end: int) -> np.ndarray:
    canvas.update()
    canvas.postscript(file="temp.ps", colormode="color", height=dim, width=dim)
    im = Image.open("temp.ps").convert("L")
    im_get_start: int = box_start + 10
    im_get_end: int = box_end - 10
    im = Image.fromarray(np.array(im)[im_get_start:im_get_end, im_get_start:im_get_end])
    im = im.resize((28, 28))
    im_np = 255 - np.array(im)
    im_np = 1.0 * (im_np / 255 > 0.01)

    return im_np


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--nnet', type=str, required=True, help="")
    parser.add_argument('--img', type=str, default="", help="")
    parser.add_argument('--dim', type=int, default=500, help="")
    parser.add_argument('--show', action='store_true', default=False, help="")
    parser.add_argument('--cvae', action='store_true', default=False, help="")
    args = parser.parse_args()

    # load nnet
    encoder: nn.Module = get_encoder_variational(args.cvae)
    encoder.load_state_dict(torch.load(f"{args.nnet}/encoder.pt"))
    decoder: nn.Module = get_decoder(args.cvae)
    decoder.load_state_dict(torch.load(f"{args.nnet}/decoder.pt"))

    nnet: nn.Module = get_ae(encoder, decoder, args.cvae)
    nnet.eval()

    line_ids = []

    # load image
    if len(args.img) > 0:
        im = Image.open(args.img).convert("L")
        # im = im.resize((28, 28), resample=PIL.Image.BOX)
        im = im.resize((28, 28))
        im_np = np.array(im)

        im_np = 1.0 * (im_np/255 > 0.01)

        # print_nnet_output(nnet, im_np)
    else:
        root = tk.Tk()

        box_start: int = int(args.dim * 0.1)
        box_end: int = args.dim - box_start
        canvas = Canvas(root, bg='white', width=args.dim, height=args.dim)
        expln_entry = tk.Entry(root)

        def run(event):
            im_canvas_np: np.ndarray = get_canvas_img(canvas, args.dim, box_start, box_end)
            if args.show:
                pylab.imshow(im_canvas_np, cmap="gray")
                pylab.show()

            nnet_out = run_nnet(nnet, im_canvas_np)

            # im_np = np.array(im)
            pylab.imshow(nnet_out.reshape((28, 28)), cmap="gray")
            pylab.show()

        def undo(event):
            if len(line_ids) > 0:
                line_id_pop = line_ids.pop(-1)
                canvas.delete(line_id_pop)

        def clear(event):
            canvas.delete('all')

            buttonBG = canvas.create_rectangle(0, 0, 100, 30, fill="grey40", outline="grey60")
            buttonTXT = canvas.create_text(50, 15, text="run")
            canvas.tag_bind(buttonBG, "<Button-1>", run)
            canvas.tag_bind(buttonTXT, "<Button-1>", run)

            buttonBG = canvas.create_rectangle(105, 0, 205, 30, fill="grey40", outline="grey60")
            buttonTXT = canvas.create_text(155, 15, text="clear")
            canvas.tag_bind(buttonBG, "<Button-1>", clear)
            canvas.tag_bind(buttonTXT, "<Button-1>", clear)

            canvas.create_window(310, 15, window=expln_entry)
            buttonBG = canvas.create_rectangle(400, 0, 500, 30, fill="grey40", outline="grey60")
            buttonTXT = canvas.create_text(450, 15, text="explain")
            # canvas.tag_bind(buttonBG, "<Button-1>", explain)
            # canvas.tag_bind(buttonTXT, "<Button-1>", explain)

            canvas.create_line(box_start, box_start, box_start, box_end, fill="black")
            canvas.create_line(box_start, box_end, box_end, box_end, fill="black")
            canvas.create_line(box_start, box_start, box_end, box_start, fill="black")
            canvas.create_line(box_end, box_start, box_end, box_end, fill="black")

            buttonBG = canvas.create_rectangle(0, box_end + 10, 100, box_end + 40, fill="grey40", outline="grey60")
            buttonTXT = canvas.create_text(50, box_end + 25, text="undo")
            canvas.tag_bind(buttonBG, "<Button-1>", undo)
            canvas.tag_bind(buttonTXT, "<Button-1>", undo)

            canvas.pack()

        clear(None)

        # draw
        line_points = []
        line_options = {'fill': 'black', 'width': int(np.ceil(10 * args.dim/500))}

        def draw_line(event):
            global line_id
            line_points.extend((event.x, event.y))
            if line_id is not None:
                canvas.delete(line_id)
            line_id = canvas.create_line(line_points, **line_options)
            line_ids.append(line_id)

        def set_start(event):
            line_points.extend((event.x, event.y))

        def end_line(event=None):
            global line_id
            line_points.clear()
            line_id = None

        # Tk
        canvas.bind('<Button-1>', set_start)
        canvas.bind('<B1-Motion>', draw_line)
        canvas.bind('<ButtonRelease-1>', end_line)
        root.mainloop()


if __name__ == "__main__":
    main()
