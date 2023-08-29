from warnings import warn
from typing import Optional, Callable, Dict, Any, List, Sequence
from itertools import chain
import os.path as osp
from glob import glob

from PIL import Image
from torch.utils.data import Dataset


class PromptingImageFolderDataset(Dataset):
    def __init__(
        self, 
        data_dir: str = "data",
        img_exts: Sequence[str] = ['.png', '.jpg'],
        transform: Optional[Callable] = None
    ):
        super().__init__()
        self.data_dir = data_dir
        self.img_exts = img_exts
        self.transform = transform

        self.__data = self.__prepare()

    def __prepare(self) -> List[Dict[str, Any]]:
        data = dict()

        for img_path in chain(*[glob(f"*{img_ext}", root_dir=self.data_dir) for img_ext in self.img_exts]):
            img_id = osp.splitext(img_path)[0]
            img_path = osp.join(self.data_dir, img_path)
            data[img_id] = dict(img_path=img_path)
        
        for prompt_path in glob("*.txt", root_dir=self.data_dir):
            img_id = osp.splitext(prompt_path)[0]
            prompt_path = osp.join(self.data_dir, prompt_path)
            if img_id in data:
                with open(prompt_path, 'r', encoding='utf8') as f:
                    data[img_id]['prompt'] = f.read()
            else:
                warn(f"Prompt {img_id} does not associated with any images!")

        remove_ids = []
        for img_id in data:
            if 'prompt' not in data[img_id]:
                warn(f"Image {img_id} does not have prompt. Removing...")
                remove_ids.append(img_id)
            if 'img_path' not in data[img_id]:
                warn(f"Prompt {img_id} does not associated with any images. Removing...")
                remove_ids.append(img_id)
        for img_id in remove_ids:
            data.pop(img_id)
        
        return list(data.values())
    
    def __len__(self):
        return len(self.__data)
    
    def __getitem__(self, idx: int):
        img_path, prompt = self.__data[idx].values()

        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        
        return {"image": img, "prompt": prompt}
