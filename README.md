
# GRCF: Geometrical Relation-aware  Multi-modal Network with  Confidence Fusion for Text-based Image Captioning
## Introduction:
Pytorch implementation  for GRCF.  
  
## Pretrained GRCF model:
We release the following pretrained GRCF model  for the TextCaps dataset:
description | download link | validation set | test set|
:---:  | :---: | :---: | :---:|
GRCF best |[Baidu Netdisk](https://pan.baidu.com/s/1DDW7ev4v9VVkdWz4u5wDmg) code: `ampz`|`BLEU-4` 25.7, `CIDEr` 106.9 | `BLEU-4` 21.0, `CIDEr` 96.6 |

## Installation:
Our implementation is based on [mmf](https://github.com/facebookresearch/mmf) framework, and and built upon [M4C-Captioner](https://github.com/ronghanghu/mmf/tree/project/m4c_captioner_pre_release/projects/M4C_Captioner). Please refer to [mmf's document](https://mmf.sh/docs/) for more details on installation requirements.
## Dataset:
  (1) The original Textcaps dataset is from https://textvqa.org/textcaps/dataset/.  Please download them from the links below and extract them under dataname  directory:
  
 *  [object Faster R-CNN Features of TextCaps](https://dl.fbaipublicfiles.com/pythia/features/open_images.tar.gz)
  
 *  [OCR Faster R-CNN Features of TextCaps](https://dl.fbaipublicfiles.com/pythia/m4c/data/m4c_textvqa_ocr_en_frcn_features.tar.gz)
 
 *  [detectron weights of TextCaps](http://dl.fbaipublicfiles.com/pythia/data/detectron_weights.tar.gz)
  
  
  (2) We use [CNMT's imdb](https://github.com/wzk1015/CNMT) file  to build our model.
  
  * imdb_train.npy
  
  * imdb_val_filtered_by_image_id.npy
  
  * imdb_test_filtered_by_image_id.npy
  
  (3) At last, our data directory (*/home/`username`/.cache/torch/mmf/data/datasets/*) structure should look like this:
  
  textcaps
  
  >defaults

  >>detectron

  >>extras

  dataname
  
  >m4c_textvqa_ocr_en_frcn_features
  
  >open_images
  
  >>detectron_fix_100
  
  >imdb
  
  >>imdb_train.npy
  
  >>imdb_val_filtered_by_image_id.npy
  
  >>imdb_test_filtered_by_image_id.npy
  
## Running the code:

*  to train the grmncf model on the TextCaps training set to get `best.model`:

```bash
CUDA_VISIBLE_DEVICES=0,1 mmf_run datasets=cnmtdata \
    model=grcf \
    config=projects/grcf/configs/grcf_defaults.yaml \
    env.save_dir=./save/grcf/defaults \
    run_type=train_val   
```

 * Using `best.model` to generate prediction json files on the validation set:
 
 ```bash
 CUDA_VISIBLE_DEVICES=1 mmf_predict datasets=cnmtdata \
    model=grcf \
    config=projects/grcf/configs/grcf_defaults.yaml \
    env.save_dir=./save/grcf/defaults \
    run_type=val \
    checkpoint.resume_file=./save/grcf/defaults/best.model
  ```
  
* Using `best.model` to generate prediction json files on the test set:

 ```bash
 CUDA_VISIBLE_DEVICES=1 mmf_predict datasets=cnmtdata \
    model=grcf \
    config=projects/grcf/configs/grcf_defaults.yaml \
    env.save_dir=./save/grcf/defaults \
    run_type=test \
    checkpoint.resume_file=./save/grcf/defaults/best.model
  ```
* to evaluate the prediction `json file` of the TextCaps validation set:

```bash
python /home/zhangsm/Python_project/mmf/projects/m4c_captioner/scripts/textcaps_eval.py \
    --set val \
    --annotation_file /home/zhangsm/.cache/torch/mmf/data/datasets/textcaps/defaults/annotations/imdb_val.npy \
    --pred_file   json_file
```
* You can submit the JSON file of the TextCaps test set to the EvalAI server for the result.
## Annotation:
python=3.7.0

pytorch=1.6.0

huggingface-hub=0.2.1

Some important files' paths are as follows:
file name | path | description | 
:---:  | :---: | :---: | 
grcf_defaults.yaml | projects/grcf/configs/| | 
defaults.yaml|mmf/configs/datasets/cnmtdata/||
grcf.py|mmf/models/||
builder.py, dataset.py|mmf/datasets/builders/cnmtdata/||
losses.py, metrics.py|mmf/modules/||
