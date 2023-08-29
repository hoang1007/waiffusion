from typing import Tuple, Union
import torchvision.transforms.functional as F

from PIL.Image import Image

class Resize:
    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        keep_ratio: bool = False,
        padding: bool = True,
        pad_val: int = 0,
        padding_mode: str = "constant",
    ):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

        self.keep_ratio = keep_ratio
        self.padding = padding
        self.pad_val = pad_val
        self.padding_mode = padding_mode

    def __call__(self, img):
        if self.keep_ratio:
            if isinstance(img, Image):
                src_width, src_height = img.size
            else:
                src_height, src_width = img.shape

            dst_height, dst_width = self.size
            scale_factor = min(dst_width / src_width, dst_height / src_height)
            scaled_height, scaled_width = (
                int(src_height * scale_factor),
                int(src_width * scale_factor),
            )
            img = F.resize(img, (scaled_height, scaled_width))

            if self.padding:
                pad_left = (dst_width - scaled_width) // 2
                pad_top = (dst_height - scaled_height) // 2
                pad_right = dst_width - scaled_width - pad_left
                pad_bot = dst_height - scaled_height - pad_top
                img = F.pad(
                    img,
                    padding=(pad_left, pad_top, pad_right, pad_bot),
                    fill=self.pad_val,
                    padding_mode=self.padding_mode,
                )
        else:
            img = F.resize(img, self.size)

        return img
