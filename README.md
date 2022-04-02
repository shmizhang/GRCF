
# GRMNCF: Geometrical Relation-aware  Multi-modal Network with  Confidence Fusion for Text-based Image Captioning
## Introduction:
Pytorch implementation  for GRMNCF.  
  
## Installation:
Our implementation is based on [mmf](https://github.com/facebookresearch/mmf) framework, and and built upon [M4C-Captioner](https://github.com/ronghanghu/mmf/tree/project/m4c_captioner_pre_release/projects/M4C_Captioner). Please refer to [mmf's document](https://mmf.sh/docs/) for more details on installation requirements.
## Dataset:
  The original Textcaps dataset is from https://textvqa.org/textcaps/dataset/. Following [CNMT's dataset](https://github.com/wzk1015/CNMT), we use it to build our model.
  At last, our data directory (*/home/username/.cache/torch/mmf/data/datasets/cnmt_data/*) structure should look like this:
  |-open_images
  
  |-----detectron_fix_100
  
  |-m4c_textvqa_ocr_en_frcn_features
  
  |-imdb
  
  |-----
  
  |-----
  
## Running the code:
   
## Annotation:
  split_dataset.py:                   split the train, valid and test sets.     
  vae.py: bayesian Learning part.  
  Yelp_social_relation.py (Equation.11 Fij):  compute the  the friendship similarity in Yelp dataset.  
  geographical_correlation_level.py (Equation.12):  compute the geographical correlation level between POIs.
