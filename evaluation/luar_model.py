import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from transformers import AutoModel
import math

from metrics import retrieval


class SelfAttention(nn.Module):
    """Implements Dot-Product Self-Attention as used in "Attention is all You Need".
    """
    def __init__(self):
        super(SelfAttention, self).__init__()

    def forward(self, k, q, v):
        d_k = q.size(-1)
        scores = torch.matmul(k, q.transpose(-2, -1)) / math.sqrt(d_k)
        p_attn = F.softmax(scores, dim=-1)

        return torch.matmul(p_attn, v)

class Transformer(nn.Module):
    """Defines the UAR Transformer-based model.
    """
    def __init__(self):
        super(Transformer, self).__init__()

        self.create_transformer()
        self.attn_fn = SelfAttention()
        self.linear = nn.Linear(768, 512)
        
    def create_transformer(self):
        """Creates the transformer model.
        """
        model_path = os.path.join(transformer_path, "paraphrase-distilroberta-base-v1")
        self.transformer = AutoModel.from_pretrained(model_path)

    def mean_pooling(self, token_embeddings, attention_mask):
        """Mean Pooling as described in the SBERT paper.
        """
        input_mask_expanded = repeat(attention_mask, 'b l -> b l d', d=768).float()
        sum_embeddings = reduce(token_embeddings * input_mask_expanded, 'b l d -> b d', 'sum')
        sum_mask = torch.clamp(reduce(input_mask_expanded, 'b l d -> b d', 'sum'), min=1e-9)
        return sum_embeddings / sum_mask

    def get_author_embedding(
        self, 
        text, 
        output_hidden_states=False, 
        output_attentions=False
    ):
        """Computes the Author Embedding. 
        """
        input_ids, attention_mask = text[0], text[1]
        B, N, E, _ = input_ids.shape
        
        input_ids = rearrange(input_ids, 'b n e l -> (b n e) l')
        attention_mask = rearrange(attention_mask, 'b n e l -> (b n e) l')

        outputs = self.transformer(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True, 
        )
        
        # calculate comment embedding:
        comment_embedding = self.mean_pooling(outputs['last_hidden_state'], attention_mask)
        comment_embedding = rearrange(comment_embedding, '(b n e) l -> (b n) e l', b=B, n=N, e=E)

        # calculate episode embedding:
        episode_embedding = reduce(self.attn_fn(comment_embedding, comment_embedding, comment_embedding), 
                                   'b e l -> b l', 'max')
        episode_embedding = self.linear(episode_embedding)

        out = {
            "episode_embedding": episode_embedding,
            "comment_embedding": comment_embedding,
        }

        if output_hidden_states:
            # skip embedding layer
            out["hidden_states"] = torch.stack(outputs["hidden_states"][1:])
        if output_attentions:
            out["attentions"] = torch.stack(outputs["attentions"])

        return out
    
    def forward(
        self, 
        data,
        output_hidden_states=False, 
        output_attentions=False
    ):
        """Calculates a fixed-length feature vector for a batch of episode samples.
        """
        output = self.get_author_embedding(
            data,
            output_hidden_states,
            output_attentions
        )

        return output

