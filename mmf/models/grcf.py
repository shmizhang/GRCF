# Copyright (c) Facebook, Inc. and its affiliates.
import functools
import logging
import math

import torch
import torch.nn.functional as F
from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from mmf.modules.layers import ClassifierLayer
from mmf.utils.build import build_image_encoder
from omegaconf import OmegaConf
from torch import nn
from transformers.modeling_bert import (
    BertConfig,
    BertEmbeddings,
    BertEncoder,
    BertPreTrainedModel,
)
from mmf.modules.cnmt_encoders import ImageEncoder
from sklearn.metrics.pairwise import rbf_kernel
from mmf.modules.refine_mmt import Refine_MMT
logger = logging.getLogger(__name__)
import numpy as np 
import scipy.sparse as sp
from scipy.sparse import csr_matrix

@registry.register_model("grcf")
class MM4C(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.mmt_config = BertConfig(**self.config.mmt)
        self.graph_config = BertConfig(**self.config.global_graph)
        self._datasets = registry.get("config").datasets.split(",")
        self.coverage_ratio = -1e10
    @classmethod
    def config_path(cls):
        return "configs/models/mm4c/defaults.yaml"

    def build(self):
        # modules requiring custom learning rates (usually for finetuning)
        self.finetune_modules = []

        # split model building into several components
        self._build_obj_encoding()
        self._build_ocr_encoding()
        self._build_transformer1()
        self._build_mmt()
        self._build_output()
        self._build_vocab_dict()
        
    def _build_vocab_dict(self):
        self.vocab_dict = {}
        model_data_dir=self.config["model_data_dir"]
        with open("/home/zhangsm/.cache/torch/mmf/data/datasets/cnmt_data/vocab_textcap_threshold_10.txt") as f:
            lines = f.readlines()
            for i in range(len(lines)):
                self.vocab_dict[lines[i][:-1]] = i
                
    def _build_encoder_config(self):
        return OmegaConf.create(
            {
                "type": "finetune_faster_rcnn_fpn_fc7",
                "params": {
                    "in_dim": 2048,
                    "weights_file": "models/detectron.defaults/fc7_w.pkl",
                    "bias_file": "models/detectron.defaults/fc7_b.pkl",
                    "model_data_dir": self.config.model_data_dir,
                },
            }
        )

    def _build_obj_encoding(self):
        self.obj_faster_rcnn_fc7 = ImageEncoder(
            encoder_type='finetune_faster_rcnn_fpn_fc7',
            in_dim=2048,
            weights_file='detectron/fc6/fc7_w.pkl',
            bias_file='detectron/fc6/fc7_b.pkl',
            model_data_dir="/home/zhangsm/.cache/torch/mmf/data/datasets/textcaps/defaults/"
        )
        # apply smaller lr to pretrained Faster R-CNN fc7
        self.finetune_modules.append(
            {"module": self.obj_faster_rcnn_fc7, "lr_scale": 0.1}
        )
        self.linear_obj_feat_to_mmt_in = nn.Linear(
            self.config.obj.mmt_in_dim, self.mmt_config.hidden_size
        )

        # object location feature: relative bounding box coordinates (4-dim)
        self.linear_obj_bbox_to_mmt_in = nn.Linear(4, self.mmt_config.hidden_size)

        self.obj_feat_layer_norm = nn.LayerNorm(self.mmt_config.hidden_size)
        self.obj_bbox_layer_norm = nn.LayerNorm(self.mmt_config.hidden_size)
        self.obj_drop = nn.Dropout(self.config.obj.dropout_prob)

    def _build_ocr_encoding(self):
  
        self.ocr_faster_rcnn_fc7 = ImageEncoder(
            encoder_type='finetune_faster_rcnn_fpn_fc7',
            in_dim=2048,
            weights_file='detectron/fc6/fc7_w.pkl',
            
            
            bias_file='detectron/fc6/fc7_b.pkl',
            model_data_dir="/home/zhangsm/.cache/torch/mmf/data/datasets/textcaps/defaults/"
        )
        #self.config.lr_scale_frcn
        self.finetune_modules.append(
            {"module": self.ocr_faster_rcnn_fc7, "lr_scale": 0.1}
        )

        self.linear_ocr_feat_to_mmt_in = nn.Linear(
            self.config.ocr.mmt_in_dim, self.mmt_config.hidden_size
        )

        # OCR location feature: relative bounding box coordinates (4-dim)
        self.linear_ocr_bbox_to_mmt_in = nn.Linear(4, self.mmt_config.hidden_size)
        self.linear_ocr_conf_to_mmt_in = nn.Linear(1,self.mmt_config.hidden_size)

        self.ocr_feat_layer_norm = nn.LayerNorm(self.mmt_config.hidden_size)
        self.ocr_bbox_layer_norm = nn.LayerNorm(self.mmt_config.hidden_size)
        self.ocr_conf_layer_norm = nn.LayerNorm(self.mmt_config.hidden_size)
        self.ocr_drop = nn.Dropout(self.config.ocr.dropout_prob)
        
    def _build_transformer1(self):
         self.transformer1 = Transformer1(self.graph_config)
         
    def _build_mmt(self):
        self.mmt = MMT(self.mmt_config)
        
        self.finetune_modules.append(
            {"module": self.mmt, "lr_scale": self.config.lr_scale_mmt}
        )
  

    def _build_output(self):
        # dynamic OCR-copying scores with pointer network
        self.ocr_ptr_net = OcrPtrNet(**self.config.classifier.ocr_ptr_net)
        # fixed answer vocabulary scores
        #self.ptr_attention_net = OcrPtrNet(**self.config.classifier.ocr_ptr_net)
        num_choices = registry.get(self._datasets[0] + "_num_final_outputs")
        
        self.classifier = nn.Linear(self.mmt_config.hidden_size,6736)
  
        self.vocab_size = num_choices
        self.answer_processor = registry.get(
            self._datasets[0] + "_answer_processor"
        )

    def forward(self, sample_list):
        # fwd_results holds intermediate forward pass results
        # TODO possibly replace it with another sample list
        fwd_results = {}
        self._forward_obj_encoding(sample_list, fwd_results)
        self._forward_ocr_encoding(sample_list, fwd_results)
        self._forward_relation_ocr_and_obj(sample_list, fwd_results)
        self._forward_transformer1(sample_list, fwd_results)
        self._forward_mmt_and_output(sample_list, fwd_results)


        results = {"scores": fwd_results["scores"],
          "ocr_mask": fwd_results["ocr_mask"],
          "relation_ocr":fwd_results["relation_ocr"],
          "relation_obj":fwd_results["relation_obj"],
          "ocr_emb":fwd_results["ocr_emb"],
          "obj_emb":fwd_results["obj_emb"],
          "conf1_pred":fwd_results["conf1_pred"],
          "conf2_pred":fwd_results["conf2_pred"],} 
        return results


    def _forward_obj_encoding(self, sample_list, fwd_results):
        # object appearance feature: Faster R-CNN fc7
        obj_fc6 = sample_list.image_feature_0
        obj_fc7 = self.obj_faster_rcnn_fc7(obj_fc6)
        obj_fc7 = F.normalize(obj_fc7, dim=-1)

        obj_feat = obj_fc7
        obj_bbox = sample_list.obj_bbox_coordinates

        obj_mmt_in = self.obj_feat_layer_norm(
            self.linear_obj_feat_to_mmt_in(obj_feat)
        ) + self.obj_bbox_layer_norm(self.linear_obj_bbox_to_mmt_in(obj_bbox))
        
        obj_mmt_in = self.obj_drop(obj_mmt_in)
        fwd_results["obj_mmt_in"] = obj_mmt_in
        # binary mask of valid object vs padding
        obj_nums = sample_list.image_info_0.max_features
        fwd_results["obj_mask"] = _get_mask(obj_nums, obj_mmt_in.size(1))

    def _forward_ocr_encoding(self, sample_list, fwd_results):

        # OCR FastText feature (300-dim)
        ocr_fasttext = sample_list.context_feature_0
        ocr_fasttext = F.normalize(ocr_fasttext, dim=-1)
        assert ocr_fasttext.size(-1) == 300

        # OCR PHOC feature (604-dim)
        ocr_phoc = sample_list.context_feature_1
        ocr_phoc = F.normalize(ocr_phoc, dim=-1)
        assert ocr_phoc.size(-1) == 604

        # OCR appearance feature: Faster R-CNN fc7
        ocr_fc6 = sample_list.image_feature_1[:, : ocr_fasttext.size(1), :]
        ocr_fc7 = self.ocr_faster_rcnn_fc7(ocr_fc6)
        ocr_fc7 = F.normalize(ocr_fc7, dim=-1)

        # OCR order vectors (legacy from LoRRA model; set to all zeros)
        # TODO remove OCR order vectors; they are not needed
        ocr_order_vectors = torch.zeros_like(sample_list.order_vectors)
 
        ocr_feat = torch.cat(
            [ocr_fasttext, ocr_phoc, ocr_fc7, ocr_order_vectors], dim=-1
        )
   
        
        ocr_bbox = sample_list.ocr_bbox_coordinates

        ocr_conf = sample_list.ocr_confidence
        ocr_mmt_in = self.ocr_feat_layer_norm(
            self.linear_ocr_feat_to_mmt_in(ocr_feat)
        )  + self.ocr_conf_layer_norm(self.linear_ocr_conf_to_mmt_in(ocr_conf)) + self.ocr_bbox_layer_norm(self.linear_ocr_bbox_to_mmt_in(ocr_bbox))
        
        ocr_mmt_in = self.ocr_drop(ocr_mmt_in)
        fwd_results["ocr_mmt_in"] = ocr_mmt_in
        # binary mask of valid OCR vs padding
        ocr_nums = sample_list.context_info_0.max_features
        fwd_results["ocr_mask"] = _get_mask(ocr_nums, ocr_mmt_in.size(1))


        # multiple_list helps to mask out the same word from different sources (vocab and OCR, or different OCR)
        num_samples = len(sample_list.ocr_token)
        self.ocr_to_vocab = torch.ones((num_samples,50), device=torch.device("cuda"), dtype=torch.int64) * self.answer_processor.UNK_IDX
        #[b_s, 50]
        #[b_s, 6786]
        self.multiple_list = [[[] for i in range(6786)] for j  in range(num_samples)]
        #[6736]
        for i in range(num_samples):
            for j in range(len(sample_list.ocr_token[i])):
                token = sample_list.ocr_token[i][j]
                if token in self.vocab_dict.keys():
                    vidx = self.vocab_dict[token]
                    self.ocr_to_vocab[i][j] = vidx
                    self.multiple_list[i][vidx].append(6736+j)
        for i in range(num_samples):
            for j in range(len(self.vocab_dict)):
                for k in self.multiple_list[i][j]:
                    self.multiple_list[i][k].extend(self.multiple_list[i][j])  
            
        self.ocr_to_vocab.unsqueeze_(1)
        #[b_s, 30, 50]
        self.ocr_to_vocab = self.ocr_to_vocab.repeat((1, 30, 1))
        
    def cal_place_pairwise_dist(self,place_coordinates):
    # this method calculates the pair-wise rbf distance
        gamma = 60
        p_correlation = rbf_kernel(place_coordinates, gamma=gamma)
        np.fill_diagonal(p_correlation, 0)
        return p_correlation
    def compute_relation_adj(self,relation_obj,fwd_results):
        relation_obj = sp.coo_matrix(relation_obj) + sp.eye(relation_obj.shape[0])
        row_sum=np.array(relation_obj.sum(1))
        row_sum[row_sum==0]=1
        d_inv_sqrt = np.power(row_sum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        result=d_mat_inv_sqrt.dot(relation_obj).dot(d_mat_inv_sqrt)
 
        return torch.Tensor(result.toarray())
        
    def _forward_relation_ocr_and_obj(self, sample_list, fwd_results):
        #[b_s,100,4]
        obj_bbox = sample_list.obj_bbox_coordinates.cpu()
        #[b_s,50,4]
        ocr_bbox = sample_list.ocr_bbox_coordinates.cpu()

        obj_adj_list=[]
        ocr_adj_list=[]
        for i in range(obj_bbox.shape[0]):
            obj_tmp=self.cal_place_pairwise_dist(obj_bbox[i])
            obj_adj=self.compute_relation_adj(obj_tmp,fwd_results)
            obj_adj_list.append(obj_adj)
            
            ocr_tmp=self.cal_place_pairwise_dist(ocr_bbox[i])
            ocr_adj=self.compute_relation_adj(ocr_tmp,fwd_results)
            ocr_adj_list.append(ocr_adj)
            
        #[b_s, 50, 50]
        ocr_relation=torch.cat(ocr_adj_list,dim=0).reshape(ocr_bbox.shape[0],ocr_bbox.shape[1],ocr_bbox.shape[1])
        ocr_relation=ocr_relation.cuda()
        fwd_results["relation_ocr"]=ocr_relation

        #[b_s, 100, 100]
        obj_relation=torch.cat(obj_adj_list,dim=0).reshape(obj_bbox.shape[0],obj_bbox.shape[1],obj_bbox.shape[1])
        obj_relation=obj_relation.cuda()
        fwd_results["relation_obj"]=obj_relation
     
    def _forward_transformer1(self, sample_list, fwd_results):
      
        visual_emb, text_emb,obj_emb,ocr_emb_gcn,confidence1_score, confidence2_score = self.transformer1(
            text_emb=fwd_results['ocr_mmt_in'], text_mask=fwd_results['ocr_mask'], 
            visual_emb=fwd_results['obj_mmt_in'], visual_mask=fwd_results['obj_mask'],obj_relation=fwd_results["relation_obj"],
            ocr_relation=fwd_results["relation_ocr"],
    
        )
        fwd_results['conf1_pred'] = confidence1_score
        fwd_results['conf2_pred'] = confidence2_score * fwd_results['ocr_mask']
        fwd_results['obj_mmt_in'] = visual_emb
        fwd_results['ocr_mmt_in'] = text_emb    
        fwd_results['obj_emb'] =obj_emb   
        fwd_results['ocr_emb'] = ocr_emb_gcn

    def _forward_mmt(self, sample_list, fwd_results):
   

        mmt_results = self.mmt(
            obj_emb=fwd_results['obj_mmt_in'],
            obj_mask=fwd_results['obj_mask'],
            ocr_emb=fwd_results['ocr_mmt_in'],
            ocr_mask=fwd_results['ocr_mask'],
            fixed_ans_emb=self.classifier.weight,
            prev_inds=fwd_results['prev_inds'],
        )

        fwd_results.update(mmt_results)

        
    def _forward_output(self, sample_list, fwd_results,flag):

        mmt_dec_output = fwd_results['mmt_dec_output']
        mmt_ocr_output = fwd_results['mmt_ocr_output']
        ocr_mask = fwd_results['ocr_mask']
        
        #[b_s, 30, 6736]
        fixed_scores = self.classifier(mmt_dec_output)
        #print(fixed_scores.size())
        #[b_s, 30, 768], [b_s, 50, 768],[b_s, 50]
        #[b_s, 30, 50]
        dynamic_ocr_scores, _ = self.ocr_ptr_net(
            mmt_dec_output, mmt_ocr_output, ocr_mask
        )

        #[b_s, 30, 6736]=[b_s, 30, 6736],  [b_s, 30, 50], [b_s, 30, 50]
        fixed_scores.scatter_add_(2, self.ocr_to_vocab, dynamic_ocr_scores)
        scores = torch.cat([fixed_scores, dynamic_ocr_scores], dim=-1) # (b_s,30,6786)
        fwd_results['scores'] = scores
        fwd_results["scores"][..., self.answer_processor.UNK_IDX] = -1e10

    def _forward_mmt_and_output(self, sample_list, fwd_results):
        if self.training:
            fwd_results["prev_inds"] = sample_list.train_prev_inds.clone()
            self._forward_mmt(sample_list, fwd_results)
            self._forward_output(sample_list, fwd_results,1)
        else:
            dec_step_num = sample_list.train_prev_inds.size(1)
            # fill prev_inds with BOS_IDX at index 0, and zeros elsewhere
            #[b_s, 30]
            fwd_results["prev_inds"] = torch.zeros_like(sample_list.train_prev_inds)
            fwd_results["prev_inds"][:, 0] = self.answer_processor.BOS_IDX
      
            # greedy decoding at test time
            for _ in range(dec_step_num):
                self._forward_mmt(sample_list, fwd_results)
                self._forward_output(sample_list, fwd_results,1)
                geo_score= torch.zeros(fwd_results["scores"].shape, device=fwd_results["scores"].device)
                repetition_mask = torch.zeros(fwd_results["scores"].shape, device=fwd_results["scores"].device) # (b_s,30,6786)
                geo_ocr_score,_=self.ocr_ptr_net(
                             fwd_results['mmt_dec_output'], fwd_results['ocr_emb'], fwd_results['ocr_mask'])
                #[0,b_s-1]
                for sample_idx in range(fwd_results["scores"].shape[0]):
                    #[0,29]
                    for step in range(fwd_results["scores"].shape[1]):
                        wd = fwd_results["prev_inds"][sample_idx][step]
                        if wd <= 20:    #common words
                            continue
                        repetition_mask[sample_idx, step+1:, wd] =  self.coverage_ratio
                        for same_wd in self.multiple_list[sample_idx][wd]:
                            repetition_mask[sample_idx, step+1:, same_wd] =  self.coverage_ratio
                        if wd <6736: #from the fixed vocabulary
                           continue

                        geo_score[sample_idx,step+1:,6736:]+= geo_ocr_score[sample_idx,step+1:]
                fwd_results["scores"] = fwd_results["scores"] + repetition_mask +0.0005*geo_score  
                # find the highest scoring output (either a fixed vocab
                # or an OCR), and add it to prev_inds for auto-regressive
                # decoding
                
                argmax_inds = fwd_results["scores"].argmax(dim=-1)
                fwd_results["prev_inds"][:, 1:] = argmax_inds[:, :-1]

    def get_optimizer_parameters(self, config):
        optimizer_param_groups = []

        base_lr = config.optimizer.params.lr
        # collect all the parameters that need different/scaled lr
        finetune_params_set = set()
        for m in self.finetune_modules:
            optimizer_param_groups.append(
                {
                    "params": list(m["module"].parameters()),
                    "lr": base_lr * m["lr_scale"],
                }
            )
            finetune_params_set.update(list(m["module"].parameters()))
        # remaining_params are those parameters w/ default lr
        remaining_params = [
            p for p in self.parameters() if p not in finetune_params_set
        ]
        # put the default lr parameters at the beginning
        # so that the printed lr (of group 0) matches the default lr
        optimizer_param_groups.insert(0, {"params": remaining_params})

        return optimizer_param_groups

    @classmethod
    def update_registry_for_pretrained(cls, config, checkpoint, full_output):
        from omegaconf import OmegaConf

        # Hack datasets using OmegaConf
        datasets = full_output["full_config"].datasets
        dataset = datasets.split(",")[0]
        config_mock = OmegaConf.create({"datasets": datasets})
        registry.register("config", config_mock)
        registry.register(
            f"{dataset}_num_final_outputs",
            # Need to add as it is subtracted
            checkpoint["classifier.module.weight"].size(0)
            + config.classifier.ocr_max_num,
        )
        # Fix this later, when processor pipeline is available
        answer_processor = OmegaConf.create({"BOS_IDX": 1})
        registry.register(f"{dataset}_answer_processor", answer_processor)

class MMT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.prev_pred_embeddings = PrevPredEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.init_weights()

    def forward(
        self,
        obj_emb,
        obj_mask,
        ocr_emb,
        ocr_mask,
        fixed_ans_emb,
        prev_inds,
    ):

        # build embeddings for predictions in previous decoding steps
        # fixed_ans_emb is an embedding lookup table for each fixed vocabulary
        #[6736, 768],  [b_s, 50, 768],  [b_s, 30]= [b_s,30,768 ]
        #print(prev_inds)
        dec_emb = self.prev_pred_embeddings(fixed_ans_emb, ocr_emb, prev_inds)
        # a zero mask for decoding steps, so the encoding steps elements can't
        # attend to decoding steps.
        # A triangular causal mask will be filled for the decoding steps
        # later in extended_attention_mask
        dec_mask = torch.zeros(
            dec_emb.size(0), dec_emb.size(1), dtype=torch.float32, device=dec_emb.device
        )
        ##encoder_inputs = torch.cat([txt_emb, obj_emb, ocr_emb, dec_emb], dim=1)
        encoder_inputs = torch.cat([obj_emb, ocr_emb, dec_emb], dim=1)
        ##attention_mask = torch.cat([txt_mask, obj_mask, ocr_mask, dec_mask], dim=1)
        attention_mask = torch.cat([ obj_mask, ocr_mask, dec_mask], dim=1)

        # offsets of each modality in the joint embedding space
        obj_max_num = obj_mask.size(-1)
        ocr_max_num = ocr_mask.size(-1)
        dec_max_num = dec_mask.size(-1)
        ocr_begin =obj_max_num
        ocr_end = ocr_begin + ocr_max_num

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, from_seq_length, to_seq_length]
        # So we can broadcast to
        # [batch_size, num_heads, from_seq_length, to_seq_length]
        to_seq_length = attention_mask.size(1)
        from_seq_length = to_seq_length

        # generate the attention mask similar to prefix LM
        # all elements can attend to the elements in encoding steps
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.repeat(
            1, 1, from_seq_length, 1
        )
        # decoding step elements can attend to themselves in a causal manner
        extended_attention_mask[:, :, -dec_max_num:, -dec_max_num:] = _get_causal_mask(
            dec_max_num, encoder_inputs.device
        )

        # flip the mask, so that invalid attention pairs have -10000.
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers
        encoder_outputs = self.encoder(
            encoder_inputs, extended_attention_mask, head_mask=head_mask
        )

        mmt_seq_output = encoder_outputs[0]
        mmt_ocr_output = mmt_seq_output[:, ocr_begin:ocr_end]
        mmt_dec_output = mmt_seq_output[:, -dec_max_num:]
        results = {
            "mmt_seq_output": mmt_seq_output,
            "mmt_ocr_output": mmt_ocr_output,
            "mmt_dec_output": mmt_dec_output,
        }
        return results    

class PrevPredEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        MAX_DEC_LENGTH = 100
        MAX_TYPE_NUM = 5
        hidden_size = config.hidden_size
        ln_eps = config.layer_norm_eps

        self.position_embeddings = nn.Embedding(MAX_DEC_LENGTH, hidden_size)
        self.token_type_embeddings = nn.Embedding(MAX_TYPE_NUM, hidden_size)

        self.ans_layer_norm = nn.LayerNorm(hidden_size, eps=ln_eps)
        self.ocr_layer_norm = nn.LayerNorm(hidden_size, eps=ln_eps)
        self.emb_layer_norm = nn.LayerNorm(hidden_size, eps=ln_eps)
        self.emb_dropout = nn.Dropout(config.hidden_dropout_prob)
    #[6736, 768],  [b_s, 50, 768],  [b_s, 30]
    def forward(self, ans_emb, ocr_emb, prev_inds):
        assert prev_inds.dim() == 2 and prev_inds.dtype == torch.long
        assert ans_emb.dim() == 2

        batch_size = prev_inds.size(0)
        seq_length = prev_inds.size(1)
        ans_num = ans_emb.size(0)
        # apply layer normalization to both answer embedding and OCR embedding
        # before concatenation, so that they have the same scale
        ans_emb = self.ans_layer_norm(ans_emb)
        ocr_emb = self.ocr_layer_norm(ocr_emb)
        assert ans_emb.size(-1) == ocr_emb.size(-1)
        #[b_s,6736,768]
        ans_emb = ans_emb.unsqueeze(0).expand(batch_size, -1, -1)

        #[b_s,6786,768]
        ans_ocr_emb_cat = torch.cat([ans_emb, ocr_emb], dim=1)
        #[b_s,30,768]
        raw_dec_emb = _batch_gather(ans_ocr_emb_cat, prev_inds)

        # Add position and type embedding for previous predictions
        position_ids = torch.arange(seq_length, dtype=torch.long, device=ocr_emb.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)
        position_embeddings = self.position_embeddings(position_ids)
        # Token type ids: 0 -- vocab; 1 -- OCR
        token_type_ids = prev_inds.ge(ans_num).long()
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = position_embeddings + token_type_embeddings
        embeddings = self.emb_layer_norm(embeddings)
        embeddings = self.emb_dropout(embeddings)
        dec_emb = raw_dec_emb + embeddings

        return dec_emb

#[b_s,6786,768]£¬ [b_s,30]
def _batch_gather(x, inds):
    assert x.dim() == 3
    batch_size = x.size(0)
    length = x.size(1)
    dim = x.size(2)
    #[b_s*6786,768]
    x_flat = x.view(batch_size * length, dim)
    #[0*6786, 1*6786, (b_s-1)*6786]
    batch_offsets = torch.arange(batch_size, device=inds.device) * length
    batch_offsets = batch_offsets.unsqueeze(-1)
    assert batch_offsets.dim() == inds.dim()
    #[b_s, 30]
    inds_flat = batch_offsets + inds
    #[b_s, 30, 768]
    results = F.embedding(inds_flat, x_flat)
    return results
      

class Transformer1(nn.Module):
    def __init__(self, config, image_encoder=None):
        super().__init__()
        self.t2t = Text2Text(config)
        self.gcn_ocr=GCN_Layer(768,768)
        self.gcn_obj=GCN_Layer(768,768)
        self.anchor_fc = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ELU(),
            nn.Linear(config.hidden_size // 2, 1)
            )
        self.graph_fc = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ELU(),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Sigmoid()
            )
    def forward(self, text_emb, text_mask, visual_emb, visual_mask,obj_relation,ocr_relation):
        visual_emb_gcn=self.gcn_obj(visual_emb,obj_relation)
        visual_emb=visual_emb+visual_emb_gcn
        ocr_emb_gcn=self.gcn_ocr(text_emb,ocr_relation)
        text_emb=text_emb+ocr_emb_gcn
        confidence1_score = self.anchor_fc(text_emb).squeeze(2)
        
        update_text = self.t2t(torch.cat([visual_emb, text_emb], dim=1),  torch.cat([visual_mask, text_mask], dim=1))
        visual_emb =visual_emb+torch.tanh(update_text[:, :visual_mask.size(1)])
        text_emb =  text_emb+torch.tanh(update_text[:, -text_mask.size(1):])  # [B, 50, 768]
        #visual_emb_=visual_emb
        #-0.3
        confidence2_score = self.graph_fc(text_emb).squeeze(2)
        return visual_emb, text_emb ,visual_emb_gcn, ocr_emb_gcn, confidence1_score, confidence2_score
        
class GCN_Layer(nn.Module):
    """
    A Graph Convolution Layer (GCN)
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Linear(in_features,48, bias=bias)
        self.W2 = nn.Linear(48,out_features, bias=bias)
        self.init()

    def init(self):
        stdv = 1. / math.sqrt(self.W.weight.size(1))
        stdv2 = 1. / math.sqrt(self.W2.weight.size(1))
        self.W.weight.data.uniform_(-stdv, stdv)
        self.W2.weight.data.uniform_(-stdv2, stdv2)
        
        
    def forward(self, Input,adj):
        out1=self.W(Input)
        out1=torch.relu(torch.matmul(adj,out1))
        out2=self.W2(out1)
        out2=torch.relu(torch.matmul(adj,out2))
      
        return out2       
class Text2Text(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.encoder = BertEncoder(config)
        # self.apply(self.init_weights)  # old versions of pytorch_transformers
        self.init_weights()

    def forward(self, encoder_inputs,  attention_mask):
        # generate the attention mask similar to prefix LM
        # all elements can attend to the elements in encoding steps
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.repeat(
            1, 1, attention_mask.size(1), 1
        )

        # flip the mask, so that invalid attention pairs have -10000.
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0  # 1 can attention
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs,
            extended_attention_mask,
            head_mask=head_mask
        )

        return encoder_outputs[0]

class OcrPtrNet(nn.Module):
    def __init__(self, hidden_size, query_key_size=None):
        super().__init__()

        if query_key_size is None:
            query_key_size = hidden_size
        self.hidden_size = hidden_size
        self.query_key_size = query_key_size

        self.query = nn.Linear(hidden_size, query_key_size)
        self.key = nn.Linear(hidden_size, query_key_size)
    #[b_s, 30, 768], [b_s, 50, 768],[b_s, 50]
        #[b_s, 30, 50]
    def forward(self, query_inputs, key_inputs, attention_mask):
        extended_attention_mask = (1.0 - attention_mask) * -10000.0
        assert extended_attention_mask.dim() == 2
        extended_attention_mask = extended_attention_mask.unsqueeze(1)

        query_layer = self.query(query_inputs)
        if query_layer.dim() == 2:
            query_layer = query_layer.unsqueeze(1)
            squeeze_result = True
        else:
            squeeze_result = False
        key_layer = self.key(key_inputs)

        scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        scores = scores / math.sqrt(self.query_key_size)
        scores = scores + extended_attention_mask
        if squeeze_result:
            scores = scores.squeeze(1)

        return scores, query_layer        
        
def _get_mask(nums, max_num):
    # non_pad_mask: b x lq, torch.float32, 0. on PAD
    batch_size = nums.size(0)
    arange = torch.arange(0, max_num).unsqueeze(0).expand(batch_size, -1)
    non_pad_mask = arange.to(nums.device).lt(nums.unsqueeze(-1))
    non_pad_mask = non_pad_mask.type(torch.float32)
    return non_pad_mask


@functools.lru_cache(maxsize=32)
def _get_causal_mask(seq_length, device):
    # generate a lower triangular mask
    mask = torch.zeros(seq_length, seq_length, device=device)
    for i in range(seq_length):
        for j in range(i + 1):
            mask[i, j] = 1.0
    return mask


