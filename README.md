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

## Channel Perturbation Experiment

This experiment aims to evaluate the model's reliance on signals from specific EEG channels, particularly those associated with the visual cortex. By replacing the signals from these channels with Gaussian noise, we can observe the impact on the quality of the final generated images.

**Required Scripts:**

*   `Generation/NervformerV2_insubject_retrival_perturbed_visual_feature_step1.py`: Used to extract perturbed EEG features.
*   `Generation/NervformerV2_insubject_retrival_perturbed_visual_train_dfs_step2.py`: Used to train a Diffusion Prior model adapted to the perturbed features and generate the final images.

**Execution Steps:**

**Important:** Before starting, ensure you have pre-trained `NervFormerV2` model weights (e.g., `best.pth`) available for all subjects (`sub-01` to `sub-10`) under `Retrieval/models/contrast/NervFormerV2/`. The `step1` script will attempt to load these weights. Also, verify that the `VISUAL_CORTEX_CHANNEL_INDICES` list in the `step1` script is correctly set to the indices of the channels you wish to perturb.

1.  **Step 1a: Extract Perturbed *Training Set* EEG Features**
    *   **Purpose:** Generate EEG features from the training set with noise added for all subjects. These are needed for training the Diffusion Prior model in `step2`.
    *   **Configuration:** Edit `Generation/NervformerV2_insubject_retrival_perturbed_visual_feature_step1.py` and ensure the `PROCESS_TRAIN_DATA` variable is set to `True`:
        ```python
        # --- Decide which data split to process (train or test) ---
        PROCESS_TRAIN_DATA = True # Set to True to process train data
        # --- --- ---
        ```
    *   **Execution:**
        ```bash
        cd Generation
        python NervformerV2_insubject_retrival_perturbed_visual_feature_step1.py
        cd ..
        ```
    *   **Check:** Verify that the script runs successfully for each subject and generates `..._train_perturbed_visual.pt` files in the corresponding `Generation/NervformerV2_eeg_feature/sub-XX/perturbed_visual/` directories.

2.  **Step 1b: Extract Perturbed *Test Set* EEG Features**
    *   **Purpose:** Generate EEG features from the test set with noise added for all subjects. These are needed for the final image generation in `step2`.
    *   **Configuration:** Edit `Generation/NervformerV2_insubject_retrival_perturbed_visual_feature_step1.py` and ensure the `PROCESS_TRAIN_DATA` variable is set to `False`:
        ```python
        # --- Decide which data split to process (train or test) ---
        PROCESS_TRAIN_DATA = False # Set to False to process test data
        # --- --- ---
        ```
    *   **Execution:**
        ```bash
        cd Generation
        python NervformerV2_insubject_retrival_perturbed_visual_feature_step1.py
        cd ..
        ```
    *   **Check:** Verify that the script runs successfully for each subject and generates `..._test_perturbed_visual.pt` files in the corresponding `Generation/NervformerV2_eeg_feature/sub-XX/perturbed_visual/` directories.

3.  **Step 2: Train Diffusion Prior and Generate Images**
    *   **Purpose:** (1) Train a Diffusion Prior model adapted to the perturbed data, using the perturbed training features from `sub-01`. (2) Use this trained prior model, along with each subject's perturbed test set features, to generate images.
    *   **Execution:**
        ```bash
        cd Generation
        python NervformerV2_insubject_retrival_perturbed_visual_train_dfs_step2.py
        cd ..
        ```
    *   **Note:** This step will first train the Diffusion Prior model (default 150 epochs), which may take a significant amount of time. After training, the model weights will be saved to `Generation/ckpts/diffusion_prior_All_perturbed.pt`. The script will then automatically proceed to the image generation phase.

4.  **Review Results:**
    *   The generated images will be saved in the `Generation/gen_images_perturbed_visual/` directory, organized into subdirectories by subject (e.g., `Generation/gen_images_perturbed_visual/sub-01/`, `Generation/gen_images_perturbed_visual/sub-02/`, etc.).
    *   Compare these images with the original images generated without perturbation to assess the impact of the visual cortex channel perturbation on the generation results.

**Summary:** It is crucial to run the `step1` script twice (once for the training set with `True`, once for the test set with `False`) to ensure the `step2` script has all the necessary input feature files before it can successfully execute the Diffusion Prior training and final image generation.

---

## Original Model Image Generation

This section describes how to generate images using the original EEG features (without perturbation) for the different baseline models (NervformerV2, NervformerV1, MUSE, ATMS_50).

The process involves two main steps for each model:
1.  **Extract EEG Features:** Use the corresponding `_extracted_feature_step1.py` script to process the EEG data with the pre-trained encoder for that model and save the features.
2.  **Generate Images:** Use the corresponding `_train_dfs_step2.py` script to load the extracted features and generate images, typically using a pre-trained Diffusion Prior model (`diffusion_prior_All.pt`).

**General Workflow (Repeat for each model and subject):**

**Important:** Before starting, ensure you have the appropriate pre-trained EEG encoder model weights available for the specific model and subject you want to process (e.g., under `Retrieval/models/contrast/[MODEL_NAME]/sub-XX/`). The respective `step1` scripts need these weights.

1.  **Step 1a: Extract *Training Set* Features**
    *   **Purpose:** Generate original EEG features from the training set for the chosen model and subject.
    *   **Configuration:** In the relevant `..._extracted_feature_step1.py` script (e.g., `MUSE_insubject_retrival_All_extracted_feature_step1.py`), ensure the `sub` variable is set to the desired subject ID (e.g., `sub = 'sub-02'`). Also ensure the script is configured to process the **training** data (this might involve commenting/uncommenting `EEGDataset` lines or similar logic within the script - check the script details).
    *   **Execution:**
        ```bash
        cd Generation
        python [MODEL_NAME]_insubject_retrival_All_extracted_feature_step1.py
        cd ..
        ```
        (Replace `[MODEL_NAME]` with `NervformerV2`, `NervformerV1`, `MUSE`, or `ATM_S`)
    *   **Check:** Verify that the script generates the `..._train.pt` feature file in the corresponding `Generation/[MODEL_NAME]_eeg_feature/sub-XX/` directory.

2.  **Step 1b: Extract *Test Set* Features**
    *   **Purpose:** Generate original EEG features from the test set for the chosen model and subject.
    *   **Configuration:** In the same `..._extracted_feature_step1.py` script, ensure `sub` is set correctly. Configure the script to process the **test** data (again, check the script for specific lines to modify, often related to the `EEGDataset(..., train=False)` call).
    *   **Execution:**
        ```bash
        cd Generation
        python [MODEL_NAME]_insubject_retrival_All_extracted_feature_step1.py
        cd ..
        ```
    *   **Check:** Verify that the script generates the `..._test.pt` feature file in the corresponding `Generation/[MODEL_NAME]_eeg_feature/sub-XX/` directory.

3.  **Step 2: Generate Images**
    *   **Purpose:** Load the training and test features generated in the previous steps and generate images using the pre-trained Diffusion Prior (`diffusion_prior_All.pt`).
    *   **Configuration:** In the relevant `..._train_dfs_step2.py` script (e.g., `MUSE_insubject_retrival_All_train_dfs_step2.py`), ensure the `sub` variable matches the subject processed in Step 1. Verify that the script is set up to **load** the pre-trained Diffusion Prior (`./ckpts/diffusion_prior_All.pt`) and **not** retrain it (i.e., the `pipe.train` line should be commented out).
    *   **Execution:**
        ```bash
        cd Generation
        python [MODEL_NAME]_insubject_retrival_All_train_dfs_step2.py
        cd ..
        ```
    *   **Note:** The script needs both `..._train.pt` and `..._test.pt` feature files from Step 1 to exist.

4.  **Review Results:**
    *   Generated images are typically saved in `./Generation/gen_images/` with filenames indicating the model and subject.

**Model-Specific Scripts:**

*   **NervformerV2:**
    *   Step 1: `NervformerV2_insubject_retrival_All_extracted_feature_step1.py`
    *   Step 2: `NervformerV2_insubject_retrival_All_train_dfs_step2.py` (or `_2.py`)
*   **NervformerV1:**
    *   Step 1: `NervformerV1_insubject_retrival_All_extracted_feature_step1.py`
    *   Step 2: `NervformerV1_insubject_retrival_All_train_dfs_step2.py`
*   **MUSE:**
    *   Step 1: `MUSE_insubject_retrival_All_extracted_feature_step1.py`
    *   Step 2: `MUSE_insubject_retrival_All_train_dfs_step2.py`
*   **ATMS_50 (ATM_S):**
    *   Step 1: `ATM_S_insubject_retrival_All_extracted_feature_step1.py`
    *   Step 2: `ATM_S_insubject_retrival_All_train_dfs_step2.py`

**Note on Looping:** The original scripts are often hardcoded for a single subject. To process all subjects automatically, you would need to modify the scripts to include a loop over subject IDs, similar to how the perturbed scripts were modified, ensuring paths are updated correctly within the loop. 

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

