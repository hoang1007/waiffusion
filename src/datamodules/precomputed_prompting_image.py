import os
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from .prompting_imagefolder import PromptingImageFolderDataset

from src.models.autoencoder import AutoEncoderKL
from src.models.condition_encoders import CLIPEncoder


class PrecomputedPromptingImageDataset(Dataset):
    def __init__(
        self,
        org_dataset: PromptingImageFolderDataset,
        vae: AutoEncoderKL,
        text_encoder: CLIPEncoder,
        vae_pretrained_path: str,
        device: torch.device = 'cpu',
        cache_dir: str = '.cache'
    ):
        super().__init__()

        vae = vae.load_from_checkpoint(vae_pretrained_path)

        self.cache_path = os.path.join(cache_dir, 'precomputed.pt')
        if os.path.exists(cache_dir):
            self.__precomputed_data = torch.load(self.cache_path)
        else:
            self.__precomputed_data = self.__prepare_cached(org_dataset, vae, text_encoder, device)
            
            os.makedirs(cache_dir)
            torch.save(self.__precomputed_data, self.cache_path)
    
    def __len__(self):
        return len(self.__precomputed_data)

    def __getitem__(self, idx):
        return self.__precomputed_data[idx]

    @torch.inference_mode()
    def __prepare_cached(
        self,
        dataset: PromptingImageFolderDataset,
        vae: AutoEncoderKL,
        text_encoder: CLIPEncoder,
        device: torch.device = 'cpu'
    ):
        cached_data = []
        vae.to(device)
        text_encoder.to(device)

        for item in tqdm(dataset):
            img, prompt = item['image'], item['prompt']
            img.to(device)

            # batching
            img.unsqueeze_(0)
            prompt = [prompt]

            latent = vae.encode(img).sample()
            prompt_embed = text_encoder(prompt)

            cached_data.append(dict(
                latent=latent.cpu(),
                prompt_embed=prompt_embed.cpu()
            ))
        
        return cached_data
