#test
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
# cannot run in jupyter, will meet the proxy issue
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
os.environ['http_proxy'] = 'http://10.16.35.10:13390'
os.environ['https_proxy'] = 'http://10.16.35.10:13390'
os.environ["WANDB_API_KEY"] = "KEY"
os.environ["WANDB_MODE"] = 'offline'
from itertools import combinations

# import clip
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import tqdm
from eegdatasets_leaveone_topo import EEGDataset
from eegencoder import eeg_encoder
from einops.layers.torch import Rearrange, Reduce
from lavis.models.clip_models.loss import ClipLoss
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
import random
# from utils import wandb_logger
from torch import Tensor
import math

# --- START PERTURBATION CONFIGURATION ---
# !!! IMPORTANT: Define the indices of the channels corresponding to the visual cortex !!!
# This depends on your specific EEG setup and channel ordering.
# Example: If channels 50-60 are visual cortex channels in your 63-channel setup:
VISUAL_CORTEX_CHANNEL_INDICES = [50, 56, 57, 58, 60, 61, 62] # Pz, PO3, POz, PO4, O1, Oz, O2
PERTURB_OUTPUT_DIR_SUFFIX = "perturbed_visual"
PERTURB_FILENAME_SUFFIX = "_perturbed_visual"
# --- END PERTURBATION CONFIGURATION ---


## for central
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         # Correct the calculation of div_term to match d_model dimensions properly
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         # Apply sine to even indices and cosine to odd indices
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])  # Ensure div_term is correctly aligned

#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         pe = self.pe[:x.size(0), :].unsqueeze(1).repeat(1, x.size(1), 1)
#         # Add positional encoding to input tensor
#         x = x + pe
#         return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model + 1, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term[:d_model // 2 + 1])
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])

        self.register_buffer('pe', pe)

    def forward(self, x):
        # print("x.shape: ", x.shape) # PositionalEncoding seems unused in NervFormerV2 forward path
        pe = self.pe[:x.size(0), :].unsqueeze(1).repeat(1, x.size(1), 1)
        # print("pe.shape: ", pe.shape)
        x = x + pe
        return x

class EEGAttention(nn.Module):
    def __init__(self, channel, d_model, nhead):
        super(EEGAttention, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True) # batch_first=True needed if input is [batch, seq, feature]
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.channel = channel
        self.d_model = d_model

    def forward(self, src):
        # Assuming src input is [batch_size, channel, time_length]
        # Permute to [batch_size, time_length, channel] for batch_first=True transformer
        src = src.permute(0, 2, 1)
        # src = self.pos_encoder(src) # Positional encoding might not be ideal here if channels are features
        output = self.transformer_encoder(src)
        # Permute back to [batch_size, channel, time_length]
        return output.permute(0, 2, 1)


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        # revised from shallownet
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            # nn.Conv2d(40, 40, (14, 1), (1, 1)), //only for central
            nn.Conv2d(40, 40, (63, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        # b, _, _, _ = x.shape
        # x = x.unsqueeze(1) # This unsqueeze is now done within NervFormerV2 forward
        # print("x", x.shape)
        x = self.tsconv(x)
        # print("tsconv", x.shape)
        x = self.projection(x)
        # print("projection", x.shape)
        return x


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x


class Enc_eeg(nn.Sequential):
    # This Enc_eeg seems unused by NervFormerV2, but kept for potential use by other models
    def __init__(self, emb_size=40, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            FlattenHead()
        )


class Proj_eeg(nn.Sequential):
    # This Proj_eeg seems unused by NervFormerV2, but kept for potential use by other models
    def __init__(self, embedding_dim=1440, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )

class Proj_eeg_Muse(nn.Sequential):
    # Used by NervFormerV1 and NervFormerV2
    def __init__(self, embedding_dim=36*40, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )

class Proj_img(nn.Sequential):
    # Used? Seems like identity
    def __init__(self, embedding_dim=1024, proj_dim=1024, drop_proj=0.3):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )
    def forward(self, x):
        # return x # Original was identity? Let's keep the projection layers
        return super().forward(x)


class NervFormer_EEG(nn.Module):
    def __init__(self, output_dim=1440): # output_dim=1440 matches Proj_eeg_Muse input
        super(NervFormer_EEG, self).__init__()
        self.instance_norm = nn.InstanceNorm2d(num_features=1) # Changed from InstanceNorm1d
        # self.gatnn = EEG_GAT() # GAT not used in current code
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (63, 1), (1, 1)), # Processes spatial info
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.stconv = nn.Sequential(
            nn.Conv2d(1, 40, (63, 1), (1, 1)),  # Spatial convolution first
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (1, 25), (1, 1)),  # Temporal convolution
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        # Assuming features from conv are [batch, features, seq_len_reduced]
        # Need to reshape for MultiheadAttention which expects [seq_len, batch, features] or [batch, seq_len, features] if batch_first=True
        # The output spatial dim after convs is 1x36 (check this)
        # Output of tsconv/stconv: [batch, 40, 1, 36] -> Flatten/Permute -> [36, batch, 40] or [batch, 36, 40]
        self.seq_len_after_conv = 36 # Calculate this based on conv parameters
        self.feature_dim_after_conv = 40

        # Use batch_first=True for easier handling
        self.self_attn_ts = nn.MultiheadAttention(embed_dim=self.feature_dim_after_conv, num_heads=5, batch_first=True)
        self.self_attn_st = nn.MultiheadAttention(embed_dim=self.feature_dim_after_conv, num_heads=5, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=self.feature_dim_after_conv, num_heads=8, dropout=0.75, batch_first=True)

        # Feed forward input dim needs to match flattened output after attention
        # Output after cross_attn: [batch, 2 * seq_len_after_conv, feature_dim_after_conv]
        ff_input_dim = 2 * self.seq_len_after_conv * self.feature_dim_after_conv # 2 * 36 * 40 = 2880

        self.feed_forward = nn.Sequential(
            # No need to Flatten here if input is already [batch, features_flat]
            # Restore Flatten to match the original model definition used for training
            nn.Flatten(),
            nn.Linear(ff_input_dim, 2048),   # Now Layer 1
            nn.ReLU(),                      # Now Layer 2
            nn.Dropout(0.5),                # Now Layer 3
            nn.Linear(2048, output_dim),    # Now Layer 4
        )

        # LayerNorm applied on the feature dimension
        self.norm1 = nn.LayerNorm(self.feature_dim_after_conv)
        self.norm2 = nn.LayerNorm(self.feature_dim_after_conv)
        self.norm3 = nn.LayerNorm(self.feature_dim_after_conv)


    def forward(self, x):
        # Input x expected as [batch, 1, channels, time_length] from NervFormerV2
        # print("NervFormer_EEG input x: ", x.shape) # Should be [batch, 1, 63, 250]
        x = self.instance_norm(x) # Applies normalization across channel and time dims
        # print("After instance norm: ", x.shape)

        ts_features = self.tsconv(x) # Output: [batch, 40, 1, 36]
        st_features = self.stconv(x) # Output: [batch, 40, 1, 36]
        # print("ts_features conv: ", ts_features.shape)
        # print("st_features conv: ", st_features.shape)

        # Reshape for attention: [batch, seq_len, features]
        ts_features = ts_features.squeeze(2).permute(0, 2, 1) # [batch, 36, 40]
        st_features = st_features.squeeze(2).permute(0, 2, 1) # [batch, 36, 40]
        # print("ts_features reshaped: ", ts_features.shape)
        # print("st_features reshaped: ", st_features.shape)

        # Self-attention
        bf_ts_features, _ = self.self_attn_ts(ts_features, ts_features, ts_features)
        bf_st_features, _ = self.self_attn_st(st_features, st_features, st_features)
        # print("bf_ts_features attn: ", bf_ts_features.shape) # [batch, 36, 40]
        # print("bf_st_features attn: ", bf_st_features.shape) # [batch, 36, 40]

        # Add & Norm (Residual connection)
        bf_ts_features = self.norm1(bf_ts_features + ts_features)
        bf_st_features = self.norm2(bf_st_features + st_features)
        # print("bf_ts_features normed: ", bf_ts_features.shape)
        # print("bf_st_features normed: ", bf_st_features.shape)

        # Concatenate along sequence dimension for cross-attention
        combined_features = torch.cat((bf_ts_features, bf_st_features), dim=1) # [batch, 72, 40]
        # print("combined_features cat: ", combined_features.shape)

        # Cross-attention
        # Query, Key, Value are all combined_features
        cf_combined_features, _ = self.cross_attn(combined_features, combined_features, combined_features)
        # print("cf_combined_features attn: ", cf_combined_features.shape) # [batch, 72, 40]

        # Add & Norm
        final_combined_features = self.norm3(cf_combined_features + combined_features)
        # print("final_combined_features normed: ", final_combined_features.shape) # [batch, 72, 40]

        # Flatten for feed-forward layer
        final_combined_features = final_combined_features.flatten(1) # [batch, 72 * 40] = [batch, 2880]
        # print("final_combined_features flattened: ", final_combined_features.shape)

        output_features = self.feed_forward(final_combined_features) #[batch, 1440]
        # print("output_features final: ", output_features.shape)

        return output_features


class ATM_S_insubject_retrieval_Central(nn.Module):
    # This seems unused by NervFormerV2 but kept
    def __init__(self, num_channels=14, sequence_length=250, num_subjects=1, num_features=64, num_latents=1024, num_blocks=1):
        super(ATM_S_insubject_retrieval_Central, self).__init__()
        self.attention_model = EEGAttention(num_channels, num_channels, nhead=1)
        self.subject_wise_linear = nn.ModuleList([nn.Linear(sequence_length, sequence_length) for _ in range(num_subjects)])
        self.enc_eeg = Enc_eeg() # This uses PatchEmbedding, which seems different from NervFormer's convs
        self.proj_eeg = Proj_eeg()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()

    def forward(self, x):
        x = self.attention_model(x)
        # print(f'After attention shape: {x.shape}')

        x = self.subject_wise_linear[0](x)
        # print(f'After subject-specific linear transformation shape: {x.shape}')
        eeg_embedding = self.enc_eeg(x)

        out = self.proj_eeg(eeg_embedding)
        return out

class ATM_S_insubject_retrieval_All(nn.Module):
     # This seems unused by NervFormerV2 but kept
    def __init__(self, num_channels=63, sequence_length=250, num_subjects=1, num_features=64, num_latents=1024, num_blocks=1):
        super(ATM_S_insubject_retrieval_All, self).__init__()
        self.attention_model = EEGAttention(num_channels, num_channels, nhead=1)
        self.subject_wise_linear = nn.ModuleList([nn.Linear(sequence_length, sequence_length) for _ in range(num_subjects)])
        self.enc_eeg = Enc_eeg() # Uses PatchEmbedding
        self.proj_eeg = Proj_eeg()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()

    def forward(self, x):
        x = self.attention_model(x)
        # print(f'After attention shape: {x.shape}')

        x = self.subject_wise_linear[0](x)
        # print(f'After subject-specific linear transformation shape: {x.shape}')
        eeg_embedding = self.enc_eeg(x)

        out = self.proj_eeg(eeg_embedding)
        return out

class NervFormerV1(nn.Module):
    def __init__(self, num_channels=63, sequence_length=250, num_subjects=1, num_features=64, num_latents=1024, num_blocks=1):
        super(NervFormerV1, self).__init__()
        self.enc_eeg = NervFormer_EEG() # Uses the NervFormer conv-attention block
        self.proj_eeg = Proj_eeg_Muse() # Projects the output of enc_eeg
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()

    def forward(self, x):
        # Needs input shape [batch, 1, channels, time_length] for NervFormer_EEG
        x = x.unsqueeze(1) # Add channel dimension for Conv2d
        x = self.enc_eeg(x)
        # print("enc_eegx: ", x.shape) # Output should be [batch, 1440]
        # x_reshaped = x.reshape(1024, 36*40) # Reshape is handled inside enc_eeg's feed_forward
        # print("x_reshaped: ", x_reshaped.shape) #torch.Size([1024, 1440])
        # x = x.reshape(x.size(0), -1) # Flatten is handled inside enc_eeg's feed_forward
        out = self.proj_eeg(x)
        # print("out: ", out.shape) # Should be [batch, 1024]
        return out

class NervFormerV2(nn.Module):
    def __init__(self, num_channels=63, sequence_length=250, num_subjects=1, num_features=64, num_latents=1024, num_blocks=1):
        super(NervFormerV2, self).__init__()
        # Channel attention applied first
        self.attention_model = EEGAttention(num_channels, num_channels, nhead=1)
        # Subject-specific adaptation
        self.subject_wise_linear = nn.ModuleList([nn.Linear(sequence_length, sequence_length) for _ in range(num_subjects)])
        # Main encoding block
        self.enc_eeg = NervFormer_EEG()
        # Final projection
        self.proj_eeg = Proj_eeg_Muse()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()

    def forward(self, x):
        # Input x expected as [batch, channels, time_length] e.g. [1024, 63, 250]
        # print(f'Initial input shape: {x.shape}')
        x = self.attention_model(x)
        # print(f'After attention shape: {x.shape}') # Should be [1024, 63, 250]
        x = self.subject_wise_linear[0](x)
        # print(f'After subject-specific linear transformation shape: {x.shape}') # Should be [1024, 63, 250]
        x = x.unsqueeze(1) # Add dimension for Conv2d: [1024, 1, 63, 250]
        # print(f'Shape before enc_eeg: {x.shape}')
        x = self.enc_eeg(x)
        # print("enc_eegx output: ", x.shape) # Should be [batch, 1440]
        # x_reshaped = x.reshape(1024, 36*40) # Not needed
        # print("x_reshaped: ", x_reshaped.shape) #torch.Size([1024, 1440])
        # x = x.reshape(x.size(0), -1) # Not needed
        out = self.proj_eeg(x)
        # print("out final shape: ", out.shape) # Should be [batch, 1024]
        return out

def get_eegfeatures(sub, eegmodel, dataloader, device, text_features_all, img_features_all, k,
                     perturb_channels=False, perturb_indices=None, output_dir_suffix="", filename_suffix=""):
    eegmodel.eval()
    text_features_all = text_features_all.to(device).float()
    img_features_all = img_features_all.to(device).float()
    total_loss = 0 # Loss calculation might not be relevant if only extracting features
    correct = 0    # Accuracy calculation might not be relevant
    total = 0      # Accuracy calculation might not be relevant
    alpha = 0.9
    top5_correct = 0
    top5_correct_count = 0

    all_labels = set(range(text_features_all.size(0)))
    top5_acc = 0
    mse_loss_fn = nn.MSELoss()
    ridge_lambda = 0.1
    save_features = True
    features_list = []  # List to store features
    perturbed_data_list = [] # List to store perturbed data if needed for debugging

    with torch.no_grad():
        for batch_idx, (eeg_data_orig, labels, text, text_features, img, img_features) in enumerate(tqdm.tqdm(dataloader, desc=f"Extracting Features {sub}")):
            eeg_data = eeg_data_orig.clone().to(device) # Clone original data for potential perturbation
            # print("Original eeg_data batch shape", eeg_data.shape) # [batch, channels, time_length] e.g., [1024, 63, 250]

            # --- Apply Perturbation ---
            if perturb_channels and perturb_indices is not None and len(perturb_indices) > 0:
                print(f"Applying perturbation to {len(perturb_indices)} channels...")
                for i in range(eeg_data.shape[0]): # Iterate through samples in the batch
                    for channel_idx in perturb_indices:
                        if 0 <= channel_idx < eeg_data.shape[1]: # Check index validity
                            channel_signal = eeg_data[i, channel_idx, :]
                            mu = torch.mean(channel_signal)
                            sigma = torch.std(channel_signal)
                            # Generate Gaussian noise with same shape, mean, and std
                            noise = torch.randn_like(channel_signal, device=device)
                            generated_noise = noise * sigma + mu
                            eeg_data[i, channel_idx, :] = generated_noise
                        else:
                            print(f"Warning: Channel index {channel_idx} out of bounds for data shape {eeg_data.shape}")
                # perturbed_data_list.append(eeg_data.cpu()) # Optional: store perturbed data
            # --- End Perturbation ---


            # eeg_data = eeg_data.unsqueeze(1) # Unsqueeze is handled inside model now
            text_features = text_features.to(device).float()
            labels = labels.to(device)
            img_features = img_features.to(device).float()

            # Pass potentially perturbed data to the model
            eeg_features = eegmodel(eeg_data).float() # Input shape [batch, channels, time]
            features_list.append(eeg_features.cpu()) # Move features to CPU before appending
            logit_scale = eegmodel.logit_scale

            # --- Loss and Accuracy Calculation (Optional for pure feature extraction) ---
            regress_loss =  mse_loss_fn(eeg_features, img_features)
            img_loss = eegmodel.loss_func(eeg_features, img_features, logit_scale)
            # text_loss = eegmodel.loss_func(eeg_features, text_features, logit_scale) # Text features might not be loaded
            contrastive_loss = img_loss
            loss = alpha * regress_loss *10 + (1 - alpha) * contrastive_loss*10
            total_loss += loss.item()

            for idx, label in enumerate(labels):
                possible_classes = list(all_labels - {label.item()})
                selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                selected_img_features = img_features_all[selected_classes]

                logits_img = logit_scale * eeg_features[idx] @ selected_img_features.T
                logits_single = logits_img
                predicted_label_idx = torch.argmax(logits_single).item()
                predicted_label = selected_classes[predicted_label_idx]

                if predicted_label == label.item():
                    correct += 1
                total += 1
            # --- End Optional Calculation ---

        if save_features:
            features_tensor = torch.cat(features_list, dim=0)
            print("Final features_tensor shape", features_tensor.shape)

            # Define output directory and ensure it exists
            output_base_dir = "/home/aidan/EEG_Image_decode_old/Generation/NervformerV2_eeg_feature"
            output_subdir = f"{output_base_dir}/{sub}"
            if output_dir_suffix:
                output_subdir = f"{output_subdir}/{output_dir_suffix}"
            os.makedirs(output_subdir, exist_ok=True)
            print(f"Saving features to: {output_subdir}")

            # Determine filename based on whether data was train or test
            data_split = "train" if dataloader.dataset.train else "test"
            save_path = f"{output_subdir}/NervformerV2_eeg_features_{sub}_{data_split}{filename_suffix}.pt"
            torch.save(features_tensor.cpu(), save_path) # Ensure tensor is on CPU before saving
            print(f"Saved features to {save_path}")

            # Optional: Save perturbed data for verification
            # if perturb_channels and perturb_indices:
            #     perturbed_data_tensor = torch.cat(perturbed_data_list, dim=0)
            #     perturbed_data_path = f"{output_subdir}/NervformerV2_eeg_data_{sub}_{data_split}{filename_suffix}_perturbed_input.pt"
            #     torch.save(perturbed_data_tensor.cpu(), perturbed_data_path)
            #     print(f"Saved perturbed input data to {perturbed_data_path}")


    average_loss = total_loss / (batch_idx+1) if (batch_idx+1) > 0 else 0
    accuracy = correct / total if total > 0 else 0
    return average_loss, accuracy, labels


config = {
"data_path": "/home/aidan/data/THINGSEEG/Data/Things-EEG2/Preprocessed_data_250Hz",
"project": "train_pos_img_text_rep",
"entity": "sustech_rethinkingbci",
"name": "lr=3e-4_img_pos_pro_eeg",
"lr": 3e-4,
"epochs": 50,
"batch_size": 1024,
"logger": True
}

# Process specified subject
# sub = 'sub-01' # Can be changed or looped through
subjects_to_process = [f'sub-{i:02d}' for i in range(1, 11)]
print(f"Processing subjects: {subjects_to_process}")

# Specify whether to perturb channels
DO_PERTURBATION = True # Set to True to activate perturbation


# --- Decide which data split to process (train or test) ---
# For generation, we typically want to process the TEST split.
PROCESS_TRAIN_DATA = False # Set to True to process train data for step 2 loading
# --- --- ---

# Loop through each subject
for sub in subjects_to_process:
    print(f"\n===== Processing Subject: {sub} =====")
    print(f"Perturbation active: {DO_PERTURBATION}")
    if DO_PERTURBATION:
        print(f"Perturbing channels: {VISUAL_CORTEX_CHANNEL_INDICES}")

    data_path = config['data_path']
    print("data_path: ", data_path)

    if PROCESS_TRAIN_DATA:
        print(f"Processing TRAIN data for {sub}")
        dataset_to_process = EEGDataset("all", data_path, subjects= [sub], train=True)
    else:
        print(f"Processing TEST data for {sub}")
        dataset_to_process = EEGDataset("all", data_path, subjects= [sub], train=False)

    if len(dataset_to_process) == 0:
        print(f"Warning: No data found for {sub} {'train' if PROCESS_TRAIN_DATA else 'test'} split. Skipping.")
        continue

    dataloader_to_process = DataLoader(dataset_to_process, batch_size=config["batch_size"], shuffle=False, num_workers=4) # Reduced workers

    # These features are needed for the accuracy/loss calculation within get_eegfeatures,
    # ensure they match the split being processed (train or test)
    # Handle potential missing features gracefully
    try:
        if PROCESS_TRAIN_DATA:
            text_features_all_split = dataset_to_process.text_features # Assuming these exist on train dataset object
            img_features_all_split = dataset_to_process.img_features
        else:
            text_features_all_split = dataset_to_process.text_features # Assuming these exist on test dataset object
            img_features_all_split = dataset_to_process.img_features
    except AttributeError:
         print(f"Warning: text_features or img_features not found on dataset object for {sub}. Loss/Accuracy calculation might be affected.")
         # Assign dummy tensors if needed downstream, or handle absence in get_eegfeatures
         text_features_all_split = torch.empty(0) 
         img_features_all_split = torch.empty(0)

    # Initialize the model (do this inside the loop if model architecture could change per subject, unlikely here)
    eeg_model = NervFormerV2()
    print('Number of parameters:', sum([p.numel() for p in eeg_model.parameters()]))

    # Load the pre-trained weights for the specific subject
    # Adjust the path structure and checkpoint name as needed
    # Attempt to load a specific checkpoint first, then fallback to best.pth
    model_checkpoint_path_specific = f"/home/aidan/EEG_Image_decode_old/Retrieval/models/contrast/NervFormerV2/{sub}/09-13_21-29/15.pth" # This date/time might be specific to sub-01
    model_checkpoint_path_best = f"/home/aidan/EEG_Image_decode_old/Retrieval/models/contrast/NervFormerV2/{sub}/best.pth"
    model_checkpoint_path = None

    # Prioritize specific path if it exists, otherwise use best.pth
    # Note: A better approach might be to find the *latest* checkpoint if naming is consistent.
    if os.path.exists(model_checkpoint_path_specific):
        model_checkpoint_path = model_checkpoint_path_specific
        print(f"Found specific checkpoint: {model_checkpoint_path_specific}")
    elif os.path.exists(model_checkpoint_path_best):
        model_checkpoint_path = model_checkpoint_path_best
        print(f"Found best checkpoint: {model_checkpoint_path_best}")
    else:
        print(f"Warning: Could not find model checkpoint for {sub} at {model_checkpoint_path_specific} or {model_checkpoint_path_best}. Skipping subject.")
        continue # Skip to the next subject if no checkpoint found

    print(f"Loading model weights for {sub} from: {model_checkpoint_path}")
    try:
        eeg_model.load_state_dict(torch.load(model_checkpoint_path, map_location=device)) # Use map_location
    except Exception as e:
        print(f"Error loading state dict for {sub} from {model_checkpoint_path}: {e}. Skipping subject.")
        continue
    eeg_model.to(device)

    # Run feature extraction (potentially with perturbation)
    loss, accuracy, labels = get_eegfeatures(
        sub,
        eeg_model,
        dataloader_to_process,
        device,
        text_features_all_split,
        img_features_all_split,
        k=200, # k value for accuracy calculation
        perturb_channels=DO_PERTURBATION,
        perturb_indices=VISUAL_CORTEX_CHANNEL_INDICES if DO_PERTURBATION else None,
        output_dir_suffix=PERTURB_OUTPUT_DIR_SUFFIX if DO_PERTURBATION else "",
        filename_suffix=PERTURB_FILENAME_SUFFIX if DO_PERTURBATION else ""
    )

    # Print results (accuracy/loss are less relevant if only extracting features)
    split_name = "Train" if PROCESS_TRAIN_DATA else "Test"
    perturb_status = "Perturbed" if DO_PERTURBATION else "Original"
    print(f"{sub} - {split_name} ({perturb_status}) - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")


print("\n===== Feature extraction for all subjects DONE! =====")

# No need for diffusion prior or generator imports/code in step 1
# #################
# import torch
# from torch import nn
# import torch.nn.functional as F
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
# import open_clip
# from matplotlib.font_manager import FontProperties
# import sys
# from diffusion_prior import *
# from custom_pipeline import *
# # os.environ["CUDA_VISIBLE_DEVICES"] = "5"
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# print("DONE!!!!!!") # Redundant print 