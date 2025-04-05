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
        print("x.shape: ", x.shape)
        pe = self.pe[:x.size(0), :].unsqueeze(1).repeat(1, x.size(1), 1)
        print("pe.shape: ", pe.shape)
        x = x + pe
        return x

class EEGAttention(nn.Module):
    def __init__(self, channel, d_model, nhead):
        super(EEGAttention, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.channel = channel
        self.d_model = d_model

    def forward(self, src):
        src = src.permute(2, 0, 1)  # Change shape to [time_length, batch_size, channel]
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output.permute(1, 2, 0)  # Change shape back to [batch_size, channel, time_length]

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
        x = x.unsqueeze(1)     
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
    def __init__(self, emb_size=40, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            FlattenHead()
        )

        
class Proj_eeg(nn.Sequential):
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
        return x 

class NervFormer_EEG(nn.Module):
    def __init__(self, output_dim=1440):
        super(NervFormer_EEG, self).__init__()
        self.instance_norm = nn.InstanceNorm2d(num_features=1)
        # self.gatnn = EEG_GAT()
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (63, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.stconv = nn.Sequential(
            nn.Conv2d(1, 40, (63, 1), (1, 1)),  # Spatial convolution
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (1, 25), (1, 1)),  # Temporal convolution
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.self_attn_ts = nn.MultiheadAttention(embed_dim=40, num_heads=5)
        self.self_attn_st = nn.MultiheadAttention(embed_dim=40, num_heads=5)
        self.cross_attn = nn.MultiheadAttention(embed_dim=40, num_heads=8, dropout=0.75)

        self.feed_forward = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2880, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, output_dim),
        )

        self.norm1 = nn.LayerNorm(40) #d_model=40
        self.norm2 = nn.LayerNorm(40)
        self.norm3 = nn.LayerNorm(40)


    def forward(self, x):
        x = self.instance_norm(x)
        # print("xxx: ", x.shape) # torch.Size([1024, 63, 250])
        ##### Nerv-GA#####
        # x = self.gatnn(x)
        ##################
        ts_features = self.tsconv(x)
        # print("ts_features: ", ts_features.shape)
        ts_features= ts_features.flatten(2).permute(2, 0, 1)  # [seq_len, batch, features]
        st_features = self.stconv(x).flatten(2).permute(2, 0, 1)  # [seq_len, batch, features]
        # Attention is applied over the 250 time steps. 
        bf_ts_features, _ = self.self_attn_ts(ts_features, ts_features, ts_features)
        bf_st_features, _ = self.self_attn_st(st_features, st_features, st_features)
        # LayerNorm
        bf_ts_features = self.norm1(bf_ts_features + ts_features)
        bf_st_features = self.norm2(bf_st_features + st_features)
        combined_features = torch.cat((bf_ts_features, bf_st_features), dim=0) # need to cat?
        cf_combined_features, _ = self.cross_attn(combined_features, combined_features, combined_features)
        final_combined_features = self.norm3(cf_combined_features + combined_features)
        final_combined_features = final_combined_features.permute(1, 0, 2).flatten(1)
        output_features = self.feed_forward(final_combined_features) #[1000, 1440]

        return output_features

class ATM_S_insubject_retrieval_Central(nn.Module):    
    def __init__(self, num_channels=14, sequence_length=250, num_subjects=1, num_features=64, num_latents=1024, num_blocks=1):
        super(ATM_S_insubject_retrieval_Central, self).__init__()
        self.attention_model = EEGAttention(num_channels, num_channels, nhead=1)   
        self.subject_wise_linear = nn.ModuleList([nn.Linear(sequence_length, sequence_length) for _ in range(num_subjects)])
        self.enc_eeg = Enc_eeg()
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
    def __init__(self, num_channels=63, sequence_length=250, num_subjects=1, num_features=64, num_latents=1024, num_blocks=1):
        super(ATM_S_insubject_retrieval_All, self).__init__()
        self.attention_model = EEGAttention(num_channels, num_channels, nhead=1)   
        self.subject_wise_linear = nn.ModuleList([nn.Linear(sequence_length, sequence_length) for _ in range(num_subjects)])
        self.enc_eeg = Enc_eeg()
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
        self.enc_eeg = NervFormer_EEG()
        self.proj_eeg = Proj_eeg_Muse()        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()       
        
    def forward(self, x):
        x = self.enc_eeg(x)
        # print("enc_eegx: ", x.shape) # x:  torch.Size([1024, 36, 40])
        # x_reshaped = x.reshape(1024, 36*40)
        # print("x_reshaped: ", x_reshaped.shape) #torch.Size([1024, 1440])
        x = x.reshape(x.size(0), -1)
        out = self.proj_eeg(x)
        # print("out: ", out.shape)
        return out  

class NervFormerV2(nn.Module):
    def __init__(self, num_channels=63, sequence_length=250, num_subjects=1, num_features=64, num_latents=1024, num_blocks=1):
        super(NervFormerV2, self).__init__() 
        self.attention_model = EEGAttention(num_channels, num_channels, nhead=1)   
        self.subject_wise_linear = nn.ModuleList([nn.Linear(sequence_length, sequence_length) for _ in range(num_subjects)])
        self.enc_eeg = NervFormer_EEG()
        self.proj_eeg = Proj_eeg_Muse()        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()       
        
    def forward(self, x):
        x = self.attention_model(x)
        # print(f'After attention shape: {x.shape}') #torch.Size([1024, 63, 250])
        x = self.subject_wise_linear[0](x)
        # print(f'After subject-specific linear transformation shape: {x.shape}') #torch.Size([1024, 63, 250])
        x = x.unsqueeze(1)
        x = self.enc_eeg(x)
        # print("enc_eegx: ", x.shape) # x:  torch.Size([1024, 36, 40])
        # x_reshaped = x.reshape(1024, 36*40)
        # print("x_reshaped: ", x_reshaped.shape) #torch.Size([1024, 1440])
        x = x.reshape(x.size(0), -1)
        out = self.proj_eeg(x)
        # print("out: ", out.shape)
        return out  

def get_eegfeatures(sub, eegmodel, dataloader, device, text_features_all, img_features_all, k):
    eegmodel.eval()
    text_features_all = text_features_all.to(device).float()
    img_features_all = img_features_all.to(device).float()
    total_loss = 0
    correct = 0
    total = 0
    alpha =0.9
    top5_correct = 0
    top5_correct_count = 0

    all_labels = set(range(text_features_all.size(0)))
    top5_acc = 0
    mse_loss_fn = nn.MSELoss()
    ridge_lambda = 0.1
    save_features = True
    features_list = []  # List to store features
    with torch.no_grad():
        for batch_idx, (eeg_data, labels, text, text_features, img, img_features) in enumerate(dataloader):
            eeg_data = eeg_data.to(device)
            eeg_data = eeg_data[:, :, :]
            print("eeg_data", eeg_data.shape)
            # eeg_data = eeg_data.unsqueeze(1)
            text_features = text_features.to(device).float()
            labels = labels.to(device)
            img_features = img_features.to(device).float()
            eeg_features = eegmodel(eeg_data).float()
            features_list.append(eeg_features)
            logit_scale = eegmodel.logit_scale 
                   
            regress_loss =  mse_loss_fn(eeg_features, img_features)
            # print("eeg_features", eeg_features.shape)
            # print(torch.std(eeg_features, dim=-1))
            # print(torch.std(img_features, dim=-1))
            # l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            # loss = (regress_loss + ridge_lambda * l2_norm)       
            img_loss = eegmodel.loss_func(eeg_features, img_features, logit_scale)
            text_loss = eegmodel.loss_func(eeg_features, text_features, logit_scale)
            contrastive_loss = img_loss
            # loss = img_loss + text_loss

            regress_loss =  mse_loss_fn(eeg_features, img_features)
            # print("text_loss", text_loss)
            # print("img_loss", img_loss)
            # print("regress_loss", regress_loss)            
            # l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            # loss = (regress_loss + ridge_lambda * l2_norm)       
            loss = alpha * regress_loss *10 + (1 - alpha) * contrastive_loss*10
            # print("loss", loss)
            total_loss += loss.item()
            
            for idx, label in enumerate(labels):

                possible_classes = list(all_labels - {label.item()})
                selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                selected_img_features = img_features_all[selected_classes]
                

                logits_img = logit_scale * eeg_features[idx] @ selected_img_features.T
                # logits_text = logit_scale * eeg_features[idx] @ selected_text_features.T
                # logits_single = (logits_text + logits_img) / 2.0
                logits_single = logits_img
                # print("logits_single", logits_single.shape)

                # predicted_label = selected_classes[torch.argmax(logits_single).item()]
                predicted_label = selected_classes[torch.argmax(logits_single).item()] # (n_batch, ) \in {0, 1, ..., n_cls-1}
                if predicted_label == label.item():
                    correct += 1        
                total += 1
        if save_features:
            features_tensor = torch.cat(features_list, dim=0)
            print("features_tensor", features_tensor.shape)
            # torch.save(features_tensor.cpu(), f"/home/aidan/EEG_Image_decode_old/Generation/NervformerV2_eeg_feature/{sub}/NervformerV2_eeg_features_{sub}_test.pt")  # Save features as .pt file
            torch.save(features_tensor.cpu(), f"/home/aidan/EEG_Image_decode_old/Generation/NervformerV2_eeg_feature/{sub}/NervformerV2_eeg_features_{sub}_train.pt")  # Save features as .pt file

    average_loss = total_loss / (batch_idx+1)
    accuracy = correct / total
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

sub = 'sub-01'


data_path = config['data_path']
print("data_path: ", data_path)
#test_dataset = EEGDataset("Central", data_path, subjects= ['sub-01'], train=False)
# train_dataset = EEGDataset("all", data_path, subjects= [sub], train=True)

test_dataset = EEGDataset("all", data_path, subjects= [sub], train=True)

test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)

text_features_test_all = test_dataset.text_features
img_features_test_all = test_dataset.img_features



# eeg_model = ATM_S_insubject_retrieval_All()
# eeg_model = NervFormerV1()
eeg_model = NervFormerV2()

print('number of parameters:', sum([p.numel() for p in eeg_model.parameters()]))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


eeg_model.load_state_dict(torch.load("/home/aidan/EEG_Image_decode_old/Retrieval/models/contrast/NervFormerV2/sub-01/09-13_21-29/15.pth"))
# eeg_model.load_state_dict(torch.load("/home/aidan/EEG_Image_decode_old/Retrieval/models/contrast/NervFormerV2/sub-02/09-11_05-35/15.pth"))
# eeg_model.load_state_dict(torch.load("/home/aidan/EEG_Image_decode_old/Retrieval/models/contrast/NervFormerV2/sub-03/09-09_22-09/40.pth"))
# eeg_model.load_state_dict(torch.load("/home/aidan/EEG_Image_decode_old/Retrieval/models/contrast/NervFormerV2/sub-04/09-11_05-35/20.pth"))
# eeg_model.load_state_dict(torch.load("/home/aidan/EEG_Image_decode_old/Retrieval/models/contrast/NervFormerV2/sub-05/09-12_16-12/65.pth"))
# eeg_model.load_state_dict(torch.load("/home/aidan/EEG_Image_decode_old/Retrieval/models/contrast/NervFormerV2/sub-06/09-09_22-09/135.pth"))
# eeg_model.load_state_dict(torch.load("/home/aidan/EEG_Image_decode_old/Retrieval/models/contrast/NervFormerV2/sub-07/09-11_05-35/65.pth"))
# eeg_model.load_state_dict(torch.load("/home/aidan/EEG_Image_decode_old/Retrieval/models/contrast/NervFormerV2/sub-08/09-11_05-35/20.pth"))
# eeg_model.load_state_dict(torch.load("/home/aidan/EEG_Image_decode_old/Retrieval/models/contrast/NervFormerV2/sub-09/09-11_05-35/10.pth"))
# eeg_model.load_state_dict(torch.load("/home/aidan/EEG_Image_decode_old/Retrieval/models/contrast/NervFormerV2/sub-10/09-09_22-09/20.pth"))








eeg_model.to(device)
test_loss, test_accuracy,labels = get_eegfeatures(sub, eeg_model, test_loader, device, text_features_test_all, img_features_test_all,k=200)
print(f" - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")


#################
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import open_clip
from matplotlib.font_manager import FontProperties
import sys
from diffusion_prior import *
from custom_pipeline import *
# os.environ["CUDA_VISIBLE_DEVICES"] = "5" 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


print("DONE!!!!!!")