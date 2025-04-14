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
from torch.utils.data import Dataset, DataLoader # Added Dataset import
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from PIL import Image
import os
proxy = 'http://10.16.35.10:13390'
os.environ['http_proxy'] = proxy
os.environ['https_proxy'] = proxy
os.environ["WANDB_API_KEY"] = "KEY"
os.environ["WANDB_MODE"] = 'offline'

# --- START PERTURBATION CONFIGURATION ---
PERTURB_FEATURE_DIR_SUFFIX = "perturbed_visual"
PERTURB_FEATURE_FILENAME_SUFFIX = "_perturbed_visual"
PERTURB_GEN_IMAGE_DIR = "gen_images_perturbed_visual"
PERTURB_GEN_IMAGE_FILENAME_SUFFIX = "_perturbed"
# --- END PERTURBATION CONFIGURATION ---

## Load eeg and image embeddings
# image feature (original)
emb_img_test = torch.load('/home/aidan/EEG_Image_decode_old/Generation/variables/ViT-H-14_features_test.pt')
emb_img_train = torch.load('/home/aidan/EEG_Image_decode_old/Generation/variables/ViT-H-14_features_train.pt')
print("emb_img_test (original) shape: ", emb_img_test.shape) # torch.Size([200, 1024])
print("emb_img_train (original) shape: ", emb_img_train.shape) # torch.Size([16540, 1024])

# eeg feature (PERTURBED)
# 1654clsx10imgsx4trials=66160
# sub = "sub-01"
subjects_to_process = [f'sub-{i:02d}' for i in range(1, 11)]
print(f"Processing subjects: {subjects_to_process}")

# --- Load features for Diffusion Prior Training (using sub-01 only) ---
sub_for_prior_train = 'sub-01'
print(f"\n--- Loading data for Diffusion Prior training (using {sub_for_prior_train}) ---")
feature_base_dir_prior = f'/home/aidan/EEG_Image_decode_old/Generation/NervformerV2_eeg_feature/{sub_for_prior_train}/{PERTURB_FEATURE_DIR_SUFFIX}'
emb_eeg_path_prior = f'{feature_base_dir_prior}/NervformerV2_eeg_features_{sub_for_prior_train}_train{PERTURB_FEATURE_FILENAME_SUFFIX}.pt'

if not os.path.exists(emb_eeg_path_prior):
    raise FileNotFoundError(f"Perturbed training EEG features for {sub_for_prior_train} not found at {emb_eeg_path_prior}. Please run step1 script with PROCESS_TRAIN_DATA=True first.")

print(f"Loading perturbed train EEG features for prior training from: {emb_eeg_path_prior}")
emb_eeg_prior_train = torch.load(emb_eeg_path_prior)
print("emb_eeg (perturbed train for prior) shape: ", emb_eeg_prior_train.shape)
# emb_eeg_test_path = f'{feature_base_dir}/NervformerV2_eeg_features_{sub}_test{PERTURB_FEATURE_FILENAME_SUFFIX}.pt' # Load test features inside loop

# print(f"Loading perturbed train EEG features from: {emb_eeg_path}")
# emb_eeg = torch.load(emb_eeg_path)
# print(f"Loading perturbed test EEG features from: {emb_eeg_test_path}")
# emb_eeg_test = torch.load(emb_eeg_test_path)

# print("emb_eeg (perturbed train) shape: ", emb_eeg.shape) # Should match original train shape
# print("emb_eeg_test (perturbed test) shape: ", emb_eeg_test.shape) # Should match original test shape


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

# Reshape original image features to match EEG trials
emb_img_train_4 = emb_img_train.view(1654,10,1,1024).repeat(1,1,4,1).view(-1,1024)
print("emb_img_train_4 (repeated original) shape: ", emb_img_train_4.shape) # torch.Size([66160, 1024])


dataset_prior = EmbeddingDataset(
    c_embeddings=emb_eeg_prior_train, # Use perturbed train EEG features from sub-01
    h_embeddings=emb_img_train_4, # Use original train image features (repeated)
    # h_embeds_uncond=h_embeds_imgnet
)
print("Prior Training Dataset length:", len(dataset_prior))
dataloader_prior = DataLoader(dataset_prior, batch_size=1024, shuffle=True, num_workers=16) # Reduced workers

# Initialize diffusion prior model
diffusion_prior = DiffusionPriorUNet(cond_dim=1024, dropout=0.1)
print("Diffusion Prior parameters:", sum(p.numel() for p in diffusion_prior.parameters() if p.requires_grad))
pipe = Pipe(diffusion_prior, device='cuda')

# --- Train Diffusion Prior ONCE using sub-01 data ---
# print(f"Loading pre-trained diffusion prior from: {ckpt_path}")
# if os.path.exists(ckpt_path):
#     pipe.diffusion_prior.load_state_dict(torch.load(ckpt_path, map_location=device))
# else:
#     print(f"Warning: Checkpoint {ckpt_path} not found. Diffusion prior not loaded.")

model_name = 'diffusion_prior'
print("\n--- Starting diffusion prior training (using perturbed sub-01 data) ---")
pipe.train(dataloader_prior, num_epochs=150, learning_rate=1e-3)
trained_prior_path = f'./ckpts/{model_name}_All_perturbed.pt'
torch.save(pipe.diffusion_prior.state_dict(), trained_prior_path)
print(f"Finished diffusion prior training. Saved perturbed prior to: {trained_prior_path}")
# --- Diffusion Prior is now trained (or loaded if uncommented above) ---


###########################
# Re-check proxy and CLIP model loading if needed
# proxy = 'http://10.16.35.10:13390'
# os.environ['http_proxy'] = proxy
# os.environ['https_proxy'] = proxy
cuda_device_count = torch.cuda.device_count()
print(f"CUDA devices available: {cuda_device_count}")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
# model_type = 'ViT-H-14'
# import open_clip
# vlmodel, preprocess_train, feature_extractor = open_clip.create_model_and_transforms(
#     model_type, pretrained='laion2b_s32b_b79k', precision='fp32', device = device)
# print("CLIP Model Loaded (if uncommented)")
###########################

#### Generate images using perturbed features
print("\n--- Initializing generator --- ")
generator = Generator4Embeds(num_inference_steps=4, device=device)
seed_value = 5757 # Use the same seed for comparison
print(f"Using seed: {seed_value}")

# --- Loop through subjects for IMAGE GENERATION --- 
print("\n--- Starting Image Generation Loop for Subjects ---")
for sub in subjects_to_process:
    print(f"\n===== Generating for Subject: {sub} =====")

    # Load this subject's PERTURBED TEST features
    feature_base_dir_sub = f'/home/aidan/EEG_Image_decode_old/Generation/NervformerV2_eeg_feature/{sub}/{PERTURB_FEATURE_DIR_SUFFIX}'
    emb_eeg_test_path_sub = f'{feature_base_dir_sub}/NervformerV2_eeg_features_{sub}_test{PERTURB_FEATURE_FILENAME_SUFFIX}.pt'

    if not os.path.exists(emb_eeg_test_path_sub):
        print(f"Warning: Perturbed test EEG features for {sub} not found at {emb_eeg_test_path_sub}. Skipping generation for this subject.")
        print(f"Please run step1 script with PROCESS_TRAIN_DATA=False.")
        continue

    print(f"Loading perturbed test EEG features for {sub} from: {emb_eeg_test_path_sub}")
    emb_eeg_test = torch.load(emb_eeg_test_path_sub)
    print(f"emb_eeg_test (perturbed test for {sub}) shape: ", emb_eeg_test.shape)

    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

    # Create output directory for this subject's perturbed images
    output_dir_sub = f'./{PERTURB_GEN_IMAGE_DIR}/{sub}' # Changed to save in sub-folders
    os.makedirs(output_dir_sub, exist_ok=True)
    print(f"Saving generated images for {sub} to: {output_dir_sub}")

    # from IPython.display import Image, display

    # Loop through test samples (k=0 to 199)
    for k in range(200):
        print(f"--- Processing sample k = {k} for {sub} ---", end='\r') # Use end='\r' for less verbose output

        # --- Generate from Original Image Embed (Control) ---
        # Assuming emb_img_test (original image features) is the same for all subjects
        image_embeds_orig = emb_img_test[k:k+1].to(device)
        gen = torch.Generator(device=device)
        gen.manual_seed(seed_value)
        image_from_img = generator.generate(image_embeds_orig, generator=gen)
        img_save_path = f'{output_dir_sub}/Nerv2_{sub}_generated_image_{k}_from_image_embeds_{seed_value}_ip1{PERTURB_GEN_IMAGE_FILENAME_SUFFIX}.png'
        image_from_img.save(img_save_path)

        # --- Generate Directly from Perturbed EEG Embed ---
        eeg_embeds_perturbed = emb_eeg_test[k:k+1].to(device)
        gen = torch.Generator(device=device)
        gen.manual_seed(seed_value)
        image_from_eeg = generator.generate(eeg_embeds_perturbed, generator=gen)
        eeg_save_path = f'{output_dir_sub}/Nerv2_{sub}_generated_image_{k}_from_eeg_embeds_{seed_value}_ip1{PERTURB_GEN_IMAGE_FILENAME_SUFFIX}.png'
        image_from_eeg.save(eeg_save_path)

        # --- Generate via Diffusion Prior using Perturbed EEG Embed ---
        # Uses the single prior trained earlier (pipe)
        gen = torch.Generator(device=device)
        gen.manual_seed(seed_value)
        h = pipe.generate(c_embeds=eeg_embeds_perturbed, num_inference_steps=50, guidance_scale=5.0, generator=gen)
        image_from_pipe = generator.generate(h.to(dtype=torch.float16), generator=gen)
        pipe_save_path = f'{output_dir_sub}/Nerv2_{sub}_generated_image_{k}_from_pipe_output_{seed_value}_ip1{PERTURB_GEN_IMAGE_FILENAME_SUFFIX}.png'
        image_from_pipe.save(pipe_save_path)
    print(f"\nFinished image generation loop for {sub}.")

print("\n===== Image generation for all subjects DONE! =====") 