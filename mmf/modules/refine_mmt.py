import torch
from torch import nn
import torch.nn.functional as F
import functools

from transformers.modeling_bert import (
    #BertLayerNorm,
     BertEmbeddings, BertEncoder, BertConfig,
    BertPreTrainedModel
)

class Refine_MMT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.prev_pred_embeddings = PositionEmbeddings(config)
        self.mmt_prev_pred_embeddings = PrevPredEmbeddings(config)
        self.encoder = BertEncoder(config)
        # self.apply(self.init_weights)  # old versions of pytorch_transformers
        self.init_weights()

    def forward(self,  obj_emb, obj_mask, anchor_emb, ocr_emb, ocr_mask,  fixed_ans_emb,prev_inds):
        
        
        # build embeddings for predictions in previous decoding steps
        # fixed_ans_emb is an embedding lookup table for each fixed vocabulary
        mmt_dec_emb = self.mmt_prev_pred_embeddings(fixed_ans_emb, ocr_emb, prev_inds)

        # a zero mask for decoding steps, so the encoding steps elements can't
        # attend to decoding steps.
        # A triangular causal mask will be filled for the decoding steps
        # later in extended_attention_mask
        mmt_dec_mask = torch.zeros(
            mmt_dec_emb.size(0), mmt_dec_emb.size(1), dtype=torch.float32, device=mmt_dec_emb.device
        )
        ##encoder_inputs = torch.cat([txt_emb, obj_emb, ocr_emb, dec_emb], dim=1)
        mmt_encoder_inputs = torch.cat([obj_emb, ocr_emb, mmt_dec_emb], dim=1)
        
        ##attention_mask = torch.cat([txt_mask, obj_mask, ocr_mask, dec_mask], dim=1)
        mmt_attention_mask = torch.cat([ obj_mask, ocr_mask, mmt_dec_mask], dim=1)
        # offsets of each modality in the joint embedding space
        mmt_dec_max_num = mmt_dec_mask.size(-1)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, from_seq_length, to_seq_length]
        # So we can broadcast to
        # [batch_size, num_heads, from_seq_length, to_seq_length]
        mmt_to_seq_length = mmt_attention_mask.size(1)
        mmt_from_seq_length = mmt_to_seq_length

        # generate the attention mask similar to prefix LM
        # all elements can attend to the elements in encoding steps
        mmt_extended_attention_mask = mmt_attention_mask.unsqueeze(1).unsqueeze(2)
        mmt_extended_attention_mask = mmt_extended_attention_mask.repeat(
            1, 1, mmt_from_seq_length, 1
        )
        # decoding step elements can attend to themselves in a causal manner
        mmt_extended_attention_mask[:, :, -mmt_dec_max_num:, -mmt_dec_max_num:] = \
            _get_causal_mask(mmt_dec_max_num, mmt_encoder_inputs.device)

        # flip the mask, so that invalid attention pairs have -10000.
        mmt_extended_attention_mask = (1.0 - mmt_extended_attention_mask) * -10000.0
        assert not mmt_extended_attention_mask.requires_grad
        mmt_head_mask = [None] * self.config.num_hidden_layers

        mmt_encoder_outputs = self.encoder(
            mmt_encoder_inputs,
            mmt_extended_attention_mask,
            head_mask=mmt_head_mask
        )

        seq_output = mmt_encoder_outputs[0]
        visual_output = seq_output[:, :obj_mask.size(1)]
        dec_output = seq_output[:, -mmt_dec_max_num:]
        
        
        visual_emb=visual_output
        dec_emb=dec_output
        dec_emb = self.prev_pred_embeddings(dec_emb)

        anchor_mask = torch.ones(anchor_emb.size(0), 1, dtype=torch.float32, device=anchor_emb.device)
        dec_mask = torch.zeros(
            dec_emb.size(0),
            dec_emb.size(1),
            dtype=torch.float32,
            device=dec_emb.device
        )
        encoder_inputs = torch.cat([visual_emb, anchor_emb, ocr_emb, dec_emb], dim=1)
        attention_mask = torch.cat([obj_mask, anchor_mask, ocr_mask, dec_mask], dim=1)
        
        ocr_start = obj_mask.size(1) + 1
        ocr_max_num = ocr_mask.size(-1)
        dec_max_num = dec_mask.size(-1)

        to_seq_length = attention_mask.size(1)
        from_seq_length = to_seq_length

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.repeat(
            1, 1, from_seq_length, 1
        )

        # we don't need language mask
        extended_attention_mask[:, :, -dec_max_num:, -dec_max_num:] = \
            _get_causal_mask(dec_max_num, encoder_inputs.device)

        # flip the mask, so that invalid attention pairs have -10000.
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs,
            extended_attention_mask,
            head_mask=head_mask
        )

        mmt_seq_output = encoder_outputs[0]
        mmt_ocr_output = mmt_seq_output[:, ocr_start: ocr_start+ocr_max_num]
        mmt_dec_output = mmt_seq_output[:, -dec_max_num:]

        return mmt_ocr_output, mmt_dec_output

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
        ans_emb = ans_emb.unsqueeze(0).expand(batch_size, -1, -1)
        ans_ocr_emb_cat = torch.cat([ans_emb, ocr_emb], dim=1)
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
        
class PositionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        MAX_DEC_LENGTH = 100
        hidden_size = config.hidden_size
        ln_eps = config.layer_norm_eps

        self.position_embeddings = nn.Embedding(MAX_DEC_LENGTH, hidden_size)

        self.ans_layer_norm = nn.LayerNorm(hidden_size, eps=ln_eps)
        self.emb_layer_norm = nn.LayerNorm(hidden_size, eps=ln_eps)
        self.emb_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, raw_dec_emb):
        batch_size, seq_length, _ = raw_dec_emb.size()
        dec_emb = self.ans_layer_norm(raw_dec_emb)

        position_ids = torch.arange(
            seq_length,
            dtype=torch.long,
            device=dec_emb.device
        )
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = self.emb_layer_norm(position_embeddings)
        embeddings = self.emb_dropout(embeddings)
        dec_emb = dec_emb + embeddings

        return dec_emb

@functools.lru_cache(maxsize=32)
def _get_causal_mask(seq_length, device):
    # generate a lower triangular mask
    mask = torch.zeros(seq_length, seq_length, device=device)
    for i in range(seq_length):
        for j in range(i+1):
            mask[i, j] = 1.
    return mask
    
    
def _get_mask(nums, max_num):
    # non_pad_mask: b x lq, torch.float32, 0. on PAD
    batch_size = nums.size(0)
    arange = torch.arange(0, max_num).unsqueeze(0).expand(batch_size, -1)
    non_pad_mask = arange.to(nums.device).lt(nums.unsqueeze(-1))
    non_pad_mask = non_pad_mask.type(torch.float32)
    return non_pad_mask



def _batch_gather(x, inds):
    assert x.dim() == 3
    batch_size = x.size(0)
    length = x.size(1)
    dim = x.size(2)
    x_flat = x.view(batch_size * length, dim)

    batch_offsets = torch.arange(batch_size, device=inds.device) * length
    batch_offsets = batch_offsets.unsqueeze(-1)
    assert batch_offsets.dim() == inds.dim()
    inds_flat = batch_offsets + inds
    results = F.embedding(inds_flat, x_flat)
    return results