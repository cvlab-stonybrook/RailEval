# RailEval
Deep Learning-based Rail Surface Condition Evaluation

Shilin Hu, Ke Ma, Sagnik Das, Dichang Zhang, Dimitris Samaras. ICCVW 2025.

Accepted by the 3rd workshop on Vision-based InduStrial InspectiON(VISION).

## Benchmark Dataset: TTC / Anomaly subset
Public access is pending FRA approval. The download link will be posted once approved.

For questions or early-access inquiries, contact [shilhu@cs.stonybrook.edu](mailto:shilhu@cs.stonybrook.edu).

## Getting Started
```sh
conda env create -f environment.yml
```

The code is tested with python==3.7, torch==1.12.1, and CUDA ≥ 11.3.

## Train
To train the model, run 

```sh
python train_cls.py --dataroot ./cvdata_ttc/train_X.pkl --name TTC_X --dataset_mode railnewdata --model railnewdata --checkpoints_dir $ckpt
```

**Note:** We use 4-fold cross-validation; the splits are provided in `cvdata_ttc`. Run the command for each of the four splits to reproduce the full results. The segmentation and alignment modules are pre-trained; update the paths in `models/rail_newdata.py` before training.

## Evaluate
Our checkpoints are available at [GoogleDrive](https://drive.google.com/drive/folders/1xSCAeFemBYMPoM0E13jSmitmANZTJEbD?usp=drive_link)

To test the model, run

```sh
python test_cls.py --dataroot ./cvdata_ttc/test_X.pkl --name TTC_X --dataset_mode railnewdata --model railnewdata --checkpoints_dir $ckpt --results_dir $res_dir
```

