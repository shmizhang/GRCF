
# GRMNCF: Geometrical Relation-aware  Multi-modal Network with  Confidence Fusion for Text-based Image Captioning
## Introduction:
Pytorch implementation  for GRMNCF.  
  
## Installation:
Our implementation is based on [mmf](https://github.com/facebookresearch/mmf) framework, and and built upon M4C-Captioner. Please refer to mmf's document for more details on installation requirements.
## Dataset:
  The original Textcaps dataset is from https://textvqa.org/textcaps/dataset/.
## Running the code:
   
## Annotation:
  split_dataset.py:                   split the train, valid and test sets.     
  vae.py: bayesian Learning part.  
  Yelp_social_relation.py (Equation.11 Fij):  compute the  the friendship similarity in Yelp dataset.  
  geographical_correlation_level.py (Equation.12):  compute the geographical correlation level between POIs.
