import numpy as np
import torchvision
from PIL import Image

def show_images(x, nrow=8):
    x = x * 0.5 + 0.5  # from (-1,1) to (0,1)
    grid = torchvision.utils.make_grid(x, nrow=nrow)
    arr = (grid.detach().cpu().permute(1, 2, 0).clip(0,1).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)
