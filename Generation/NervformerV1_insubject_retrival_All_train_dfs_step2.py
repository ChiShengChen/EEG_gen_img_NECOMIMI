import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import open_clip
from matplotlib.font_manager import FontProperties
import numpy as np
import sys
from diffusion_prior import *
from custom_pipeline import *
# os.environ["CUDA_VISIBLE_DEVICES"] = "5" 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from PIL import Image
import os
proxy = 'http://10.16.35.10:13390'
os.environ['http_proxy'] = proxy
os.environ['https_proxy'] = proxy
os.environ["WANDB_API_KEY"] = "KEY"
os.environ["WANDB_MODE"] = 'offline'

## Load eeg and image embeddings
# image feature
emb_img_test = torch.load('/home/aidan/EEG_Image_decode_old/Generation/variables/ViT-H-14_features_test.pt')
emb_img_train = torch.load('/home/aidan/EEG_Image_decode_old/Generation/variables/ViT-H-14_features_train.pt')
print("emb_img_test.shape: ", emb_img_test.shape) # torch.Size([200, 1024])
print("emb_img_train.shape: ", emb_img_train.shape) # torch.Size([16540, 1024])

# eeg feature
# 1654clsx10imgsx4trials=66160
sub = "sub-10"
emb_eeg = torch.load('/home/aidan/EEG_Image_decode_old/Generation/NervformerV1_eeg_feature/'+ sub +'/NervformerV1_eeg_features_'+ sub +'_train.pt')
emb_eeg_test = torch.load('/home/aidan/EEG_Image_decode_old/Generation/NervformerV1_eeg_feature/'+ sub +'/NervformerV1_eeg_features_'+ sub +'_test.pt')
print("emb_eeg.shape: ", emb_eeg.shape) # torch.Size([66160, 1024])
print("emb_eeg_test: ", emb_eeg_test.shape) # torch.Size([200, 1024])


## training prior diffusion model

class EmbeddingDataset(Dataset):

    def __init__(self, c_embeddings=None, h_embeddings=None, h_embeds_uncond=None, cond_sampling_rate=0.5):
        self.c_embeddings = c_embeddings
        self.h_embeddings = h_embeddings
        self.N_cond = 0 if self.h_embeddings is None else len(self.h_embeddings)
        self.h_embeds_uncond = h_embeds_uncond
        self.N_uncond = 0 if self.h_embeds_uncond is None else len(self.h_embeds_uncond)
        self.cond_sampling_rate = cond_sampling_rate

    def __len__(self):
        return self.N_cond

    def __getitem__(self, idx):
        return {
            "c_embedding": self.c_embeddings[idx],
            "h_embedding": self.h_embeddings[idx]
        }

emb_img_train_4 = emb_img_train.view(1654,10,1,1024).repeat(1,1,4,1).view(-1,1024)
print(emb_img_train_4.shape) # torch.Size([66160, 1024])

from torch.utils.data import DataLoader
dataset = EmbeddingDataset(
    c_embeddings=emb_eeg, h_embeddings=emb_img_train_4, 
    # h_embeds_uncond=h_embeds_imgnet
)
print(len(dataset)) #66160
dataloader = DataLoader(dataset, batch_size=1024, shuffle=True, num_workers=64)

# diffusion_prior = DiffusionPrior(dropout=0.1)
diffusion_prior = DiffusionPriorUNet(cond_dim=1024, dropout=0.1)
# number of parameters
print(sum(p.numel() for p in diffusion_prior.parameters() if p.requires_grad)) #9675648
pipe = Pipe(diffusion_prior, device='cuda')

# load pretrained model
model_name = 'diffusion_prior' # 'diffusion_prior_vice_pre_imagenet' or 'diffusion_prior_vice_pre'
pipe.diffusion_prior.load_state_dict(torch.load(f'./ckpts/{model_name}_All.pt')) #use ckpt

# pipe.train(dataloader, num_epochs=150, learning_rate=1e-3) # to 0.142  #train ckpt
# torch.save(pipe.diffusion_prior.state_dict(), f'./ckpts/{model_name}_All.pt')



###########################
import os
import torch
proxy = 'http://10.16.35.10:13390'
os.environ['http_proxy'] = proxy
os.environ['https_proxy'] = proxy
cuda_device_count = torch.cuda.device_count()
print(cuda_device_count)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# vlmodel, preprocess = clip.load("ViT-B/32", device=device)
model_type = 'ViT-H-14'
import open_clip
vlmodel, preprocess_train, feature_extractor = open_clip.create_model_and_transforms(
    model_type, pretrained='laion2b_s32b_b79k', precision='fp32', device = device)
print("PASS Proxy TEST")
###########################

#### gen image
generator = Generator4Embeds(num_inference_steps=4, device=device)
seed_value = 42


torch.manual_seed(seed_value)


if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)

# from IPython.display import Image, display
# generator = Generator4Embeds(num_inference_steps=4, device=device)


# path of ground truth: /home/ldy/Workspace/THINGS/images_set/test_images
# k = 25
# k = 61
# k = 2
for k in range(200):
    print("k: ", k)
    image_embeds = emb_img_test[k:k+1]
    print("image_embeds", image_embeds.shape) #  torch.Size([1, 1024])

    gen = torch.Generator(device=device)
    gen.manual_seed(seed_value)
    image = generator.generate(image_embeds, generator=gen)
    # display(image)
    image.save(f'./gen_images/Nerv1_sub10_generated_image_{k}_from_image_embeds.png')


    gen.manual_seed(seed_value)
    print("emb_eeg_guide: ", emb_eeg_test[k:k+1].shape) # torch.Size([1, 1024])
    image = generator.generate(emb_eeg_test[k:k+1], generator=gen)
    # display(image)
    image.save(f'./gen_images/Nerv1_sub10_generated_image_{k}_from_eeg_embeds.png')


    # k = 0
    gen.manual_seed(seed_value)
    eeg_embeds = emb_eeg_test[k:k+1]
    print("image_embeds", eeg_embeds.shape)
    h = pipe.generate(c_embeds=eeg_embeds, num_inference_steps=50, guidance_scale=5.0,generator=gen)
    ## h: torch.Size([1, 1024])
    image = generator.generate(h.to(dtype=torch.float16), generator=gen)
    # display(image)
    image.save(f'./gen_images/Nerv1_sub10_generated_image_{k}_from_pipe_output.png')


