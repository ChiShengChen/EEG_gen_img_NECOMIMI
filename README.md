# NECOMIMI: Neural-Cognitive Multimodal EEG-Informed Image Generation with Diffusion Models
[![arXiv](https://img.shields.io/badge/arXiv-2410.00712-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2410.00712)  
## Abstract
NECOMIMI introduces a groundbreaking framework for generating images directly from EEG signals using advanced diffusion models. This work not only pushes the boundaries of EEG-image classification but extends into image generation, presenting a novel EEG encoder, NERV, that shows state-of-the-art performance across multiple zero-shot classification tasks.

![image](https://github.com/user-attachments/assets/df768db1-764b-4234-9630-527cb5059f32)


## Introduction
EEG has been a valuable tool in clinical settings, like diagnosing epilepsy and depression. However, with technological advancements, its application has expanded into real-time brain function analysis and now, into the challenging field of image generation from neural signals.

## Methodology
![image](https://github.com/user-attachments/assets/0db21255-20b5-481b-aefc-bb0c561f0c3c)


This paper describes a comprehensive methodology that combines EEG signal processing and diffusion models to generate images. We introduce a novel two-stage image generation process and establish the CAT Score as a new metric tailored for EEG-to-image evaluation, setting a benchmark on the ThingsEEG dataset.

![image](https://github.com/user-attachments/assets/4e507ed2-c75c-4a06-88da-7f29cf454a6f)

## Key Contributions
- Introduction of the NERV EEG encoder that demonstrates exceptional performance in EEG-based image generation.
- Development of a two-stage generative framework that enhances image quality and semantic accuracy.
- Proposal of the Category-based Assessment Table (CAT) Score for evaluating EEG-informed image generation.

## Experiments and Findings
Experiments demonstrate NERV's effectiveness across several zero-shot classification tasks, with a detailed exploration of the conceptual challenges in translating EEG data into precise visual representations. Despite its advancements, the generated images predominantly abstract, highlighting the inherent difficulties in processing EEG signals.





## Model Architectures

We implement and evaluate multiple model architectures:

- **NervformerV2** (NERV): Our best-performing model featuring multi-head attention with cross-attention between temporal and spatial pathways
- **NervformerV1**: A simpler version without the subject-specific attention mechanism
- **MUSE_EEG**: A baseline model using only spatial-temporal convolutions
- **ATMS_50**: A model with attention but simpler convolutional processing
- **NICE_EEG**: A baseline model with minimal processing


## Getting Started

### Environment Setup

```bash
# Create conda environment:
conda env create -f environment.yml
conda activate BCI

# Or use pip:
pip install -r requirements.txt

# Additional dependencies
pip install wandb einops open_clip_torch
pip install braindecode==0.8.1
pip install transformers==0.27.0 diffusers==0.24.0

# LAVIS installation
git clone https://github.com/salesforce/LAVIS.git
cd LAVIS
pip install -e .
```

### Training & Evaluation

#### 1. Training the EEG Encoder

```bash
cd Retrieval
python NervformerV2_insubject_retrieval.py  # Train the NervformerV2 model
```

#### 2. Image Reconstruction

```bash
cd Generation
# Step 1: Extract EEG features
python NervformerV2_insubject_retrival_All_extracted_feature_step1.py

# Step 2: Generate images from EEG features
python NervformerV2_insubject_retrival_All_train_dfs_step2.py
```

#### 3. Evaluate Metrics

```bash
cd Generation/fMRI-reconstruction-NSD/src
jupyter notebook Reconstruction_Metrics_ATM.ipynb
```

## Data Resources

The THINGS-EEG and THINGS-MEG datasets can be accessed via these sources:

- **THINGS-EEG**: [OSF Repository](https://osf.io/3jk45/)
  - Raw EEG: `eeg_dataset/raw_data/`
  - Preprocessed EEG: `eeg_dataset/preprocessed_data/`
  - Images: `image_set/`
  - DNN Features: `dnn_feature_maps/pca_feature_maps/`

- **THINGS-MEG**: [OpenNEURO Repository](https://openneuro.org/datasets/ds004212/versions/2.0.0)

- **BaiduNetDisk**: [Link](https://pan.baidu.com/s/1-1hgpoi4nereLVqE4ylE_g?pwd=nid5) (password: nid5)

## Preprocessing

If you have raw data and need to preprocess it:

```bash
# EEG preprocessing
python EEG-preprocessing/preprocessing.py

# MEG preprocessing
jupyter notebook MEG-preprocessing/pre_possess.ipynb
```

## Repository Structure

```
EEG_gen_img_NECOMIMI/
├── Retrieval/                # EEG feature extraction and retrieval experiments
│   ├── NervformerV2_insubject_retrieval.py  # Main NervformerV2 training code
│   ├── NervformerV1_insubject_retrieval.py
│   └── ...
├── Generation/               # Image generation from EEG features
│   ├── NervformerV2_insubject_retrival_All_extracted_feature_step1.py
│   ├── NervformerV2_insubject_retrival_All_train_dfs_step2.py
│   └── ...
├── EEG-preprocessing/        # Data preprocessing scripts
├── MEG-preprocessing/        # MEG preprocessing scripts
├── LAVIS/                    # External LAVIS library
├── environment.yml           # Conda environment definition
├── requirements.txt          # Pip requirements
└── README.md                 # This file
```

Most code are modified from [ATM](https://github.com/dongyangli-del/EEG_Image_decode).

## Computing Cost
For running image generation need almost 30 GB VRAM:
<img width="667" alt="image" src="https://github.com/user-attachments/assets/3f128cb2-01b1-4232-bcbb-f27709b2afcd" />


## Citation

If you use this code or find it helpful for your research, please cite our work:

```
@article{nervformer,
  title={NervformerV2: Brain Decoding and Visual Reconstruction from EEG using Advanced Attention Mechanisms},
  author={Your Name et al.},
  year={2023}
}
```

Additionally, please cite the following works that contributed to this research:

1. THINGS-EEG dataset:
```
@article{gifford2022large,
  title={A large and rich EEG dataset for modeling human visual object recognition},
  author={Gifford, Alessandro T and Dwivedi, Kshitij and Roig, Gemma and Cichy, Radoslaw M},
  journal={NeuroImage},
  year={2022}
}
```

2. THINGS-MEG dataset:
```
@article{hebart2023things,
  title={THINGS-data, a multimodal collection of large-scale datasets for investigating object representations in human brain and behavior},
  author={Hebart, Martin N and Contier, Oliver and Teichmann, Lina and Rockter, Adam H and Zheng, Charles Y and Kidder, Alexis and Corriveau, Anna and Vaziri-Pashkam, Maryam and Baker, Chris I},
  journal={eLife},
  year={2023}
}
```

3. Data preprocessing methodology:
```
@article{song2023decoding,
  title={Decoding Natural Images from EEG for Object Recognition},
  author={Song, Yonghao and Liu, Bingchuan and Li, Xiang and Shi, Nanlin and Wang, Yijun and Gao, Xiaorong},
  journal={arXiv preprint arXiv:2308.13234},
  year={2023}
}
```
4. This paper:
```
@article{chen2024necomimi,
  title={NECOMIMI: Neural-Cognitive Multimodal EEG-informed Image Generation with Diffusion Models},
  author={Chen, Chi-Sheng},
  journal={arXiv preprint arXiv:2410.00712},
  year={2024}
}
```

