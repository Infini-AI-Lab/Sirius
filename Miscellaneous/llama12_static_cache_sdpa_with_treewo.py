# Adapted from Huggingface Llama File 

''' 
SIRIUS addition 

The CUDA GRAPH version of tree building script 
The performance is slightly worse than torch.compile 
'''

from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from transformers.activations import ACT2FN
import math
from transformers.models.llama.modeling_llama import(
    LlamaRMSNorm,
    LlamaConfig,
    PreTrainedModel,
    apply_rotary_pos_emb,
    ACT2FN
)

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

from transformers.models.llama.modeling_llama import (
LlamaConfig, LlamaRotaryEmbedding, LlamaLinearScalingRotaryEmbedding, LlamaDynamicNTKScalingRotaryEmbedding, 
apply_rotary_pos_emb, repeat_kv 
) 
# from transformers.cache_utils import Cache, StaticCache, SinkCache 
from transformers.cache_utils import Cache, SinkCache 
from cache import StaticCacheSDPA 
import torch.nn.functional as F 
import numpy as np 

from transformers.generation.logits_process import (
    LogitsProcessorList, 
) 

from transformers.generation.stopping_criteria import (
    StoppingCriteriaList, 
    validate_stopping_criteria, 
) 

from transformers.generation.utils import GenerateOutput, GenerateDecoderOnlyOutput, GenerateNonBeamOutput, GenerateEncoderDecoderOutput

from time import time 
from termcolor import colored 

from cache import StaticCache2 
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )

def select_neurons(neuron_stat, method, k):
    if method == 'topk':
        weight, indices = torch.topk(neuron_stat, k, dim=-1)
    elif method == 'topk_sample':
        topk_weight, topk_indices = torch.topk(neuron_stat, k // 2, dim=-1)
        neuron_stat_clone = neuron_stat.clone()
        neuron_stat_clone.scatter_(index=topk_indices, dim=1, value=0)
        sampled_indices = torch.multinomial(neuron_stat_clone, k // 2, replacement=False)
        indices = torch.cat((topk_indices, sampled_indices), dim=-1)
        weight = torch.cat((topk_weight, torch.gather(neuron_stat, 1, sampled_indices)), dim=-1)
    elif method == 'sample':
        indices = torch.multinomial(neuron_stat, k, replacement=False)
        weight = torch.gather(neuron_stat, 1, indices)
    elif method == 'random': 
        indices = torch.multinomial(torch.ones_like(neuron_stat), k, replacement=False)
        weight = torch.gather(neuron_stat, 1, indices)
    else:
        raise NotImplementedError

    return weight, indices
def get_llama_griffin(model,  k_schedule):
    config = model.config
    for i, l in enumerate(model.model.layers):
        new_mlp = GriffinLlamaMLP(config, k_schedule[i])
        new_mlp.gate_proj = l.mlp.gate_proj
        new_mlp.up_proj = l.mlp.up_proj
        new_mlp.down_proj = l.mlp.down_proj
        new_mlp.act_fn = l.mlp.act_fn

        if config.selection_method == 'magnitude':
            assert k_schedule[i] > 0.0
            gate_stat = l.mlp.gate_proj.weight.data.norm(dim=1)
            up_stat = l.mlp.up_proj.weight.data.norm(dim=1)
            stat = (gate_stat * up_stat).unsqueeze(0)
            _, indices = torch.topk(stat, int(stat.shape[1] * new_mlp.k_factor), dim=-1)
            new_mlp.prepare_reduced_weights(indices)
            new_mlp.mag_mask = torch.ones(stat.shape[-1], dtype=bool)
            new_mlp.mag_mask[indices[0]] = False

        l.mlp = new_mlp
    
    return model 

class LlamaMLPnotnorma(nn.Module): 
    def __init__(self, config, k_factor):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size 
        print("hidden_size {} intermediate_size {}".format(self.hidden_size, self.intermediate_size / 2)) 
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size // 2, bias = False) 
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size // 2, bias = False) 
        self.down_proj = nn.Linear(self.intermediate_size // 2, self.hidden_size, bias = False) 
        self.act_fn = F.silu
        
        self.k_factor = k_factor
        self.mode = config.mode
        self.inference_mode = "full"
        self.neuron_stat: torch.Tensor = None
        self.reference = 0
        assert self.inference_mode in ["full", "partial", "pass"]
        assert self.mode in ['gen', 'class'] 
        self.profilingid = 0 
    
    def forward(self, x): 
        if self.profilingid == 0: 
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)) 
        else: 
            for i in range(100): # Warmup 
                down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)) 
            torch.cuda.synchronize() 
            starttime = time() 
            for i in range(1): 
                down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)) 
            torch.cuda.synchronize() 
            endtime = time() 
            print("Time taken: {}".format(endtime - starttime)) 
            exit(0) 
        return down_proj 

def get_llama_griffin2(model,  k_schedule):
    config = model.config
    for i, l in enumerate(model.model.layers):
        new_mlp = GriffinLlamaMLP2(config, k_schedule[i]) 
        # new_mlp = LlamaMLPnotnorma(config, k_schedule).to(model.device).to(torch.bfloat16) 
        new_mlp.gate_proj = l.mlp.gate_proj 
        new_mlp.up_proj = l.mlp.up_proj 
        new_mlp.down_proj = l.mlp.down_proj 
        new_mlp.act_fn = l.mlp.act_fn 

        if config.selection_method == 'magnitude':
            assert k_schedule[i] > 0.0
            gate_stat = l.mlp.gate_proj.weight.data.norm(dim=1)
            up_stat = l.mlp.up_proj.weight.data.norm(dim=1)
            stat = (gate_stat * up_stat).unsqueeze(0)
            _, indices = torch.topk(stat, int(stat.shape[1] * new_mlp.k_factor), dim=-1)
            new_mlp.prepare_reduced_weights(indices)
            new_mlp.mag_mask = torch.ones(stat.shape[-1], dtype=bool)
            new_mlp.mag_mask[indices[0]] = False

        l.mlp = new_mlp
    
    return model

def get_llama_griffin3(model,  k_schedule):
    config = model.config
    for i, l in enumerate(model.model.layers):
        new_mlp = GriffinLlamaMLP3(config, k_schedule[i], i) 
        new_mlp.gate_proj = l.mlp.gate_proj
        new_mlp.up_proj = l.mlp.up_proj
        new_mlp.down_proj = l.mlp.down_proj
        new_mlp.act_fn = l.mlp.act_fn

        if config.selection_method == 'magnitude':
            assert k_schedule[i] > 0.0
            gate_stat = l.mlp.gate_proj.weight.data.norm(dim=1)
            up_stat = l.mlp.up_proj.weight.data.norm(dim=1)
            stat = (gate_stat * up_stat).unsqueeze(0)
            _, indices = torch.topk(stat, int(stat.shape[1] * new_mlp.k_factor), dim=-1)
            new_mlp.prepare_reduced_weights(indices)
            new_mlp.mag_mask = torch.ones(stat.shape[-1], dtype=bool)
            new_mlp.mag_mask[indices[0]] = False

        l.mlp = new_mlp
    
    return model

class GriffinLlamaMLP(nn.Module): # CATS-like 
    def __init__(self, config, k_factor):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = F.silu
        
        self.k_factor = k_factor
        self.mode = config.mode
        self.inference_mode = "full"
        self.neuron_stat: torch.Tensor = None
        assert self.inference_mode in ["full", "partial"]
        assert self.mode in ['gen', 'class']


    def prepare_reduced_weights(self, topk_indices):
        assert topk_indices.shape[0] == 1 # Batch size 1
        
        self.gate_proj_reduced = nn.Linear(self.gate_proj.weight.data.shape[1], len(topk_indices), bias=False)
        self.up_proj_reduced = nn.Linear(self.up_proj.weight.data.shape[1], len(topk_indices), bias=False)
        self.down_proj_reduced = nn.Linear(len(topk_indices), self.down_proj.weight.data.shape[0], bias=False)
        topk_indices = topk_indices[0]

        self.gate_proj_reduced.weight.data = self.gate_proj.weight.data[topk_indices]
        self.up_proj_reduced.weight.data = self.up_proj.weight.data[topk_indices]
        self.down_proj_reduced.weight.data = self.down_proj.weight.data[:, topk_indices]
    

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            k_factor = self.k_factor
            if self.mode == 'gen':
                if self.inference_mode == 'full':
                    
                    int_states :torch.Tensor = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
                        
                    down_proj = self.down_proj(int_states)

                else:
                    if k_factor == 0.0:
                        down_proj = 0 * x 
                    else: 

                        int_states :torch.Tensor = self.act_fn(self.gate_proj(x))

                        _, indices = torch.abs(int_states).topk(k=int(self.k_factor * int_states.shape[-1]), largest=False, dim=-1)
                        buffrzeros = torch.zeros_like(int_states) 
                        int_states.scatter_(dim = -1, index = indices, src = buffrzeros) 

                        int_states *= self.up_proj(x)
                        down_proj = self.down_proj(int_states)

        return down_proj
    def reset_stats(self):
        self.neuron_stat = None

class GriffinLlamaMLP2(nn.Module): # Griffin-like 
    def __init__(self, config, k_factor):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = F.silu
        
        self.k_factor = k_factor
        self.mode = config.mode
        self.inference_mode = "full"
        self.neuron_stat: torch.Tensor = None
        self.reference = 0
        assert self.inference_mode in ["full", "partial", "pass"]
        assert self.mode in ['gen', 'class']


    def prepare_reduced_weights(self, topk_indices):
        assert topk_indices.shape[0] == 1 # Batch size 1
        
        self.gate_proj_reduced = nn.Linear(self.gate_proj.weight.data.shape[1], len(topk_indices), bias=False)
        self.up_proj_reduced = nn.Linear(self.up_proj.weight.data.shape[1], len(topk_indices), bias=False)
        self.down_proj_reduced = nn.Linear(len(topk_indices), self.down_proj.weight.data.shape[0], bias=False)
        topk_indices = topk_indices[0]

        self.gate_proj_reduced.weight.data = self.gate_proj.weight.data[topk_indices]
        self.up_proj_reduced.weight.data = self.up_proj.weight.data[topk_indices]
        self.down_proj_reduced.weight.data = self.down_proj.weight.data[:, topk_indices]
    

    def forward(self, x):
            k_factor = self.k_factor
            if self.mode == 'gen':
                if self.inference_mode == 'full':
                    
                    int_states :torch.Tensor = self.act_fn(self.gate_proj(x)) * self.up_proj(x)

                    # GRIFFIN Expert Selection
                    if self.neuron_stat is None: 
                        k = int(int_states.shape[-1] * k_factor)
                        
                        states = int_states[:,self.reference:,:]
                        
                        neuron_stat = ((states / states.norm(dim=-1, keepdim=True))) # B, D
                        neuron_stat = neuron_stat.norm(dim=1)
                        
                        self.neuron_stat = neuron_stat
                        stat = self.neuron_stat
                            
                        topk_weight, topk_indices = select_neurons(stat, self.config.selection_method, k) 
                        
                        self.prepare_reduced_weights(topk_indices)
                        
                    down_proj = self.down_proj(int_states)

                elif self.inference_mode == 'partial':
                    
                    if k_factor == 0.0:
                        down_proj = 0 * x 
                    else:
                        down_proj =self.down_proj_reduced(self.act_fn(self.gate_proj_reduced(x)) * self.up_proj_reduced(x))
                else:
                    
                    int_states :torch.Tensor = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
                    down_proj = self.down_proj(int_states)


            return down_proj
    def reset_stats(self):
        self.neuron_stat = None 

class GriffinLlamaMLP3(nn.Module): # CATSTHRESHOLD-like 
    def __init__(self, config, k_factor, layer_idx): 
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size 
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False) 
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False) 
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False) 
        self.act_fn = F.silu 
        self.layer_idx = layer_idx 
        
        self.k_factor = k_factor
        self.mode = config.mode
        self.inference_mode = "full"
        self.neuron_stat: torch.Tensor = None
        assert self.inference_mode in ["full", "partial"]
        assert self.mode in ['gen', 'class']
        self.threshold_ = self.config.threshold 
        
        if self.threshold_: 
            with open(self.config.threshold_file, "rb") as f: 
                self.threshold = np.load(f)[self.layer_idx] 

    def prepare_reduced_weights(self, topk_indices):
        assert topk_indices.shape[0] == 1 # Batch size 1
        
        self.gate_proj_reduced = nn.Linear(self.gate_proj.weight.data.shape[1], len(topk_indices), bias=False)
        self.up_proj_reduced = nn.Linear(self.up_proj.weight.data.shape[1], len(topk_indices), bias=False)
        self.down_proj_reduced = nn.Linear(len(topk_indices), self.down_proj.weight.data.shape[0], bias=False)
        topk_indices = topk_indices[0]

        self.gate_proj_reduced.weight.data = self.gate_proj.weight.data[topk_indices]
        self.up_proj_reduced.weight.data = self.up_proj.weight.data[topk_indices]
        self.down_proj_reduced.weight.data = self.down_proj.weight.data[:, topk_indices]
    

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            k_factor = self.k_factor
            if self.mode == 'gen':
                if self.inference_mode == 'full':
                    
                    int_states :torch.Tensor = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
                        
                    down_proj = self.down_proj(int_states)

                else:
                    if k_factor == 0.0:
                        down_proj = 0 * x 
                    else:

                        int_states :torch.Tensor = self.act_fn(self.gate_proj(x)) # B, seq_len, D 
                        norm_ = int_states.norm(dim = -1, keepdim = True).sum(dim = 0) 
                        k = int(norm_.shape[-1] * k_factor) 
                        
                        output = norm_ > self.threshold 
                        mask = output.unsqueeze(0).expand_as(int_states) 
                        int_states = int_states * mask 
                        
                        down_proj = self.down_proj(int_states * self.up_proj(x)) 

        return down_proj
    def reset_stats(self):
        self.neuron_stat = None

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size 
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False) 
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False) 
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False) 
        self.act_fn = F.silu 

    def forward(self, x): 
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)) 

        return down_proj

class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None, 
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]: 
        bsz, q_len, _ = hidden_states.size()
        
        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        past_key_value = getattr(self, "past_key_value", past_key_value)
        cos, sin = self.rotary_emb(value_states, position_ids)
        
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
            

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value 

class LlamaSdpaAttention(LlamaAttention):

    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this if statement instead of an
        # inline conditional assignment to support both torch.compile's `dynamic=True` and `fullgraph=True`
        is_causal = True if causal_mask is None and q_len > 1 else False

        with torch.backends.cuda.sdp_kernel(enable_flash = True, enable_math = True, enable_mem_efficient = True): 
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=causal_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=is_causal, 
            ) 
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value
    
class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = (
            LlamaAttention(config=config, layer_idx=layer_idx)
        ) 

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values = None, 
        use_cache = None, 
        cache_position = None, 
        **kwargs, 
    ): 
    

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask, 
            position_ids=position_ids,
            past_key_value = past_key_values, 
            output_attentions = False, 
            use_cache = use_cache, 
            cache_position = cache_position, 
            **kwargs, 
        ) 
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,) 
        
        if use_cache: 
            outputs += (present_key_value,) 
            
        return outputs 
    
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False 
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[StaticCacheSDPA] = None, 
        use_cache: bool = None,
        cache_position: Optional[bool] = None, 
    ): 

        inputs_embeds = self.embed_tokens(input_ids) 
        
        if position_ids is None:
            # for verification
            position_ids = cache_position.unsqueeze(0) 
        hidden_states = inputs_embeds 
        
        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions = False) # for flash_attention_2, this line should be one or two if statements 

        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask, 
                position_ids=position_ids,
                past_key_values = past_key_values, 
                use_cache = use_cache, 
                cache_position = cache_position, 
            ) 

            hidden_states = layer_outputs[0] 
            
            if use_cache: 
                next_decoder_cache = layer_outputs[1] 

        hidden_states = self.norm(hidden_states)
        
        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache if isinstance(next_decoder_cache, Cache) else next_decoder_cache
            )

        return tuple([hidden_states, next_cache]) 

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        
        target_length = past_key_values.get_max_length()

        if attention_mask is not None and attention_mask.dim() == 4:
            causal_mask = attention_mask
        else:
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask

class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False) 
        self.verbose = True 
        self.generationtime = 0 
        self.totalgenerationlength = 0 
        from transformers import AutoTokenizer 
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B") 
        self.num_steps = 0 
        self.total_steps = 0 

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache = True, 
        cache_position = None, 
        past_key_values = None, 
    ): 

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache = use_cache, 
            cache_position = cache_position, 
            past_key_values = past_key_values, 
        ) 

        hidden_states = outputs[0] 
        past_key_values = outputs[1] 
        logits = self.lm_head(hidden_states)
        logits = logits.float()


        # return logits, past_key_values 
        return logits 

    def reset_states(self): 
        for layer in self.model.layers: 
            layer.mlp.reset_stats() 
    
    def set_inference_mode(self, mode): 
        for layer in self.model.layers: 
            layer.mlp.inference_mode = mode 
    
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs 
    ): 
        # this following part is for controlling the length of the input_ids 
        past_length = 0 
        cache_length = past_key_values.get_seq_length() 
        past_length = cache_length 
        print("input_ids shape {}".format(input_ids.shape)) 
        print("past_length {}".format(past_length)) 

        if input_ids.shape[1] - past_length > 1: 
            input_ids = input_ids[:, past_length :] 
            model_inputs = {"input_ids": input_ids} 
        else: 
            kwargs["executing_inputids"].copy_(input_ids[:, past_length :]) 
            model_inputs = {"input_ids": kwargs["executing_inputids"]} 
            print("executing_inputids shape {}".format(kwargs["executing_inputids"].shape)) 
        
        position_ids = kwargs.get("position_ids", None) 
        print("attention_mask shape {} sum {}".format(attention_mask.shape, attention_mask.sum(1))) 
        
        if attention_mask is not None and position_ids is None and past_length == 0: # this part is included in the original prepare_inputs_for_generation 
            # create position_ids on the fly for batch generation
            position_ids = attention_mask[:, : input_ids.shape[1]].long().cumsum(-1) - 1 
            
            position_ids.masked_fill_(attention_mask[:, : input_ids.shape[1]] == 0, 1) 
        
        # preparing cache_position 
        cache_position = kwargs.get("cache_position", None) 
        if cache_position is None: 
            cache_position = torch.arange(past_length, past_length + position_ids.shape[1], device = input_ids.device, dtype = position_ids.dtype) 
        
        position_ids = None 
        
        if past_length == 0: 
            model_inputs.update(
                { 
                    "attention_mask": attention_mask, 
                    "position_ids": position_ids, 
                    "cache_position": cache_position, 
                    "past_key_values": past_key_values, 
                } 
            ) 
        else: 
            model_inputs.update(
                {
                    "attention_mask": attention_mask, 
                    "position_ids": position_ids, 
                    "cache_position": kwargs["cache_position"], 
                    "past_key_values": past_key_values, 
                } 
            ) 
            
        return model_inputs 
    
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past 
    
    def greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        output_logits: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer = None, 
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]: 
        
        initial_len = input_ids.shape[1] 

        max_seq_len = 900 
        
        print("input_ids shape {}".format(input_ids.shape)) 
        kernel_size = self.config.kernel_size 
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList() 

        if max_length is not None:
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None

        this_peer_finished = False  # used by synced_gpus only
        initial_len = input_ids.shape[1] # prefill length 
        last_check = initial_len
        track_position_ids = None
        track_cache_position = None
        want_to_quit = False
        approve_quit = False 
        last_input_ids_print_pos = initial_len # this variable is for visualization 
        
        # shared variables for three cases 
        past_key_values = StaticCacheSDPA(config = self.config, max_batch_size = 1, max_cache_len = max_seq_len, device = input_ids.device, dtype = torch.bfloat16) 
        attention_mask = model_kwargs["attention_mask"] 
        position_ids = attention_mask[:, : input_ids.shape[1]].long().cumsum(-1) - 1 
        position_ids.masked_fill_(attention_mask[:, : input_ids.shape[1]] == 0, 1) 
        cache_position = torch.arange(0, position_ids.shape[1], device = input_ids.device, dtype = position_ids.dtype) 
        
        attention_mask_static = torch.zeros((model_kwargs["attention_mask"].shape[0], max_seq_len)).to(input_ids.device).to(model_kwargs["attention_mask"].dtype) 
        attention_mask_static[:, : model_kwargs["attention_mask"].shape[1]] = model_kwargs["attention_mask"] 
        attention_mask = attention_mask_static # attentionmask is shared 
        cumulativeprob = torch.zeros((input_ids.shape[0], self.config.treewidth)).to(input_ids.device).to(torch.float32) 
        cachepositionaddergfou = torch.arange(0, self.config.treewidth, device = input_ids.device, dtype = cache_position.dtype) 
        # prepare for attentionmask 
        min_dtype = torch.finfo(torch.bfloat16).min 
        attentionmaskgrowing = torch.full((input_ids.shape[0], 1, self.config.treewidth, max_seq_len), fill_value = 0, dtype = torch.bfloat16, device = input_ids.device) 
        attentionmaskgrowing[:, :, :, input_ids.shape[1] + 1 : ] = min_dtype 
        checkl = (self.config.kernel_size - 1) * self.config.treewidth + 1 
        attentionmaskfullgrowing = torch.full((input_ids.shape[0], 1, checkl, max_seq_len), fill_value = min_dtype, dtype = torch.bfloat16, device = input_ids.device) 
        # TODO: rewrite this part to have the triangular mask 
        attentionmaskfullgrowing = torch.triu(attentionmaskfullgrowing, diagonal = input_ids.shape[1] + 1 - (checkl - 1)) 
        selecteridx = 0 
        context_selecter = torch.zeros(((kernel_size - 1) * self.config.treewidth),).to(input_ids.device).to(torch.long) 
        
        attentionmaskidx = 0 
        initialmaskblo = torch.full((self.config.treewidth, self.config.treewidth), fill_value = min_dtype, dtype = torch.bfloat16, device = input_ids.device).fill_diagonal_(0) 
        
        # parameters for sparse gone 
        executinginputidssparse = torch.empty((1, 1)).to(input_ids.device, torch.long) 
        executingcachepositionsparse = torch.empty((1,)).to(input_ids.device, torch.long) 
        outputs = torch.empty((1, 1, self.config.vocab_size), device = input_ids.device, dtype = torch.bfloat16) 
        
        # parameters for sparse gfou 
        executinginputidssparsegfou = torch.empty((1, self.config.treewidth)).to(input_ids.device, torch.long) 
        executingpositionidssparsee = torch.empty((1, self.config.treewidth)).to(input_ids.device, torch.long) 
        executingcachepositionsparsegfou = torch.empty((self.config.treewidth,)).to(input_ids.device, torch.long) 
        outputsgfou = torch.empty((1, self.config.treewidth, self.config.vocab_size), device = input_ids.device, dtype = torch.bfloat16) 
        
        # parameters for full 
        executinginputidsfull = torch.empty((input_ids.shape[0], checkl)).to(input_ids.device, torch.long) # (kernelsize - 1) * 4 
        executingpositionidsfull = torch.empty((input_ids.shape[0], checkl)).to(input_ids.device, torch.long) 
        executingcachepositionfull = torch.empty((checkl,)).to(input_ids.device, torch.long) 
        outputsfull = torch.empty((input_ids.shape[0], checkl), device = input_ids.device, dtype = torch.bfloat16) 
        
        assert input_ids.shape[0] == 1 
        
        # first iteration 
        self.set_inference_mode("full") 
        outputs = self( # using full model 
            input_ids = input_ids, 
            attention_mask = attention_mask, 
            position_ids = None, 
            use_cache = True, 
            past_key_values = past_key_values, 
            cache_position = cache_position, 
        ) 
        next_token_logits = outputs[:, -1, :] 
        next_tokens = torch.argmax(next_token_logits, dim=-1) 
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1) 
        length_inputids = input_ids.shape[1] 
        staticinputids = torch.zeros((input_ids.shape[0], input_ids.shape[1] + 256)).to(input_ids.device).to(input_ids.dtype) 
        staticinputids[:, : input_ids.shape[1]] = input_ids 
        attention_mask[0, length_inputids - 1] = 1 
        print(colored(self.tokenizer.decode(input_ids[0, :]), "blue")) 
        
        cache_position = torch.arange(past_key_values.get_seq_length(), past_key_values.get_seq_length() + 1, device = cache_position.device, dtype = cache_position.dtype) 
        
        # preparing graph 
        executinginputidssparse.copy_(input_ids[:, -1].unsqueeze(0)) 
        executingcachepositionsparse.copy_(cache_position) 
        self.set_inference_mode("partial") 
        # warm up 
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        for i in range(100): 
            outputs = self( 
                input_ids = executinginputidssparse, 
                attention_mask = attention_mask, 
                position_ids = None, 
                use_cache = True, 
                past_key_values = past_key_values, 
                cache_position = executingcachepositionsparse, 
            ) 
            print(self.tokenizer.decode(outputs[:, -1, :].argmax(-1)[0])) 
        torch.cuda.current_stream().wait_stream(s)
        print("warming up") 
        gone = torch.cuda.CUDAGraph() 
        # getting the graph for the sparse model 
        with torch.cuda.graph(gone): 
            outputs = self( 
                input_ids = executinginputidssparse, 
                attention_mask = attention_mask, 
                position_ids = None, 
                use_cache = True, 
                past_key_values = past_key_values, 
                cache_position = executingcachepositionsparse, 
            ) 
        print(self.tokenizer.decode(outputs[:, -1, :].argmax(-1)[0])) 
        
        torch.cuda.synchronize() 
        starttime = time() 
        # getting the speed measurement 
        for i in range(10): 
            gone.replay() 
            print(self.tokenizer.decode(outputs[:, -1, :].argmax(-1)[0])) 
        torch.cuda.synchronize() 
        endtime = time() 
        print("time taken for a single forward pass {}".format((endtime - starttime)/10)) 
        
        # getting the graph for four 
        executinginputidssparsegfou.copy_(input_ids[:, -self.config.treewidth : ]) 
        executingcachepositionsparsegfou.copy_(torch.arange(past_key_values.get_seq_length() - self.config.treewidth, past_key_values.get_seq_length(), device = input_ids.device, dtype = executingcachepositionsparse.dtype)) 
        executingpositionidssparsee.copy_(executingcachepositionsparsegfou.clone().unsqueeze(0)) 
        position_marker = length_inputids 
        self.set_inference_mode("partial") 
        s = torch.cuda.Stream() 
        s.wait_stream(torch.cuda.current_stream()) 
        print("executinginputidssparsegfou shape {}".format(executinginputidssparsegfou.shape)) 
        for i in range(100): # full warm up 
            outputsgfou = self(
                input_ids = executinginputidssparsegfou, 
                attention_mask = attentionmaskgrowing, 
                position_ids = executingpositionidssparsee, 
                # position_ids = executingpositionidssparsee, 
                use_cache = True, 
                past_key_values = past_key_values, 
                cache_position = executingcachepositionsparsegfou, 
            ) 
        torch.cuda.current_stream().wait_stream(s) 
        print("warming up four") 
        gfou = torch.cuda.CUDAGraph() 
        with torch.cuda.graph(gfou): 
            outputsgfou = self(
                input_ids = executinginputidssparsegfou, 
                attention_mask = attentionmaskgrowing, 
                position_ids = executingpositionidssparsee, 
                # position_ids = executingpositionidssparsee, 
                use_cache = True, 
                past_key_values = past_key_values, 
                cache_position = executingcachepositionsparsegfou, 
            ) 
        torch.cuda.synchronize() 
        starttime = time() 
        for i in range(10): 
            gfou.replay() 
        torch.cuda.synchronize() 
        endtime = time() 
        gone.replay() 
        print(self.tokenizer.decode(outputs[:, -1, :].argmax(-1)[0])) 
        print("time taken for a single forward pass {}".format((endtime - starttime)/10)) 

        # getting the graph for the full 
        cache_positionadder = torch.arange(0, checkl, device = input_ids.device, dtype = cache_position.dtype) 
        executinginputidsfull.copy_(input_ids[:, -checkl : ]) 
        executingcachepositionfull.copy_(torch.arange(length_inputids - checkl, length_inputids, device = input_ids.device, dtype = executingcachepositionsparse.dtype)) 
        executingpositionidsfull.copy_(executingcachepositionfull.clone().unsqueeze(0)) 
        self.set_inference_mode("full") 
        # no need to adjust the attention mask 
        s = torch.cuda.Stream() 
        s.wait_stream(torch.cuda.current_stream()) 
        print("executinginputidsfull shape {}".format(executinginputidsfull.shape)) 
        for i in range(100): # full warm up 
            outputsfull = self(
                input_ids = executinginputidsfull, 
                attention_mask = attentionmaskfullgrowing, 
                position_ids = executingpositionidsfull, 
                use_cache = True, 
                past_key_values = past_key_values, 
                cache_position = executingcachepositionfull, 
            ) 
            
        torch.cuda.current_stream().wait_stream(s) 
        print("warming up full") 
        gfull = torch.cuda.CUDAGraph() 
        with torch.cuda.graph(gfull): 
            outputsfull = self(
                input_ids = executinginputidsfull, 
                attention_mask = attentionmaskfullgrowing, 
                position_ids = executingpositionidsfull, 
                use_cache = True, 
                past_key_values = past_key_values, 
                cache_position = executingcachepositionfull, 
            ) 
        torch.cuda.synchronize() 
        starttime = time() 
        for i in range(10): 
            gfull.replay() 
        torch.cuda.synchronize() 
        endtime = time() 
        gone.replay() 
        print(self.tokenizer.decode(outputs[:, -1, :].argmax(-1)[0])) 
        executinginputidsfull[:, 0 : 1] = next_tokens.item() 
        executingpositionidsfull[:, 0 : 1] = length_inputids - 1 
        attentionmaskfullgrowing.fill_(0) 
        attentionmaskfullgrowing[:, :, :, length_inputids - 1 : ] = min_dtype # resetting the attention mask 
        attentionmaskfullgrowing[:, :, 0, : length_inputids] = 0 
        # attentionmaskfullgrowing[:, :, attentionmaskidx : attentionmaskidx + self.config.treewidth, length_inputids - 1 : length_inputids - 1 + self.config.treewidth] = initialmaskblo 
        attentionmaskidx += 1 
        presenttext = [self.tokenizer.decode(next_tokens) for _ in range(self.config.treewidth)] 
        print(presenttext) 
        
        print("full time taken for a single forward pass {}".format((endtime - starttime)/10)) 
        
        starttime = time() 
        checking = True 
        
        while True: 
            if (length_inputids - last_check) != checkl: 
                checking = False 
                if (length_inputids - last_check) == 1: 
                    # one token generate four possibilities 
                    gone.replay() 
                    
                    next_token_probs = torch.log_softmax(outputs[:, -1, :]/0.6, dim = -1) # (batch_size, vocab_size) 
                    topk, topkidx = torch.topk(next_token_probs, self.config.treewidth, dim = -1) # (batch_size, treewidth) 
                    # first doesn't need filtering 
                    print(colored(self.tokenizer.decode(topkidx[0]), "cyan"), flush = True) 
                    for i in range(self.config.treewidth): 
                        presenttext[i] += self.tokenizer.decode(topkidx[0, i]) 
                    print(presenttext) 
                    
                    staticinputids[:, length_inputids : length_inputids + self.config.treewidth] = topkidx 
                    length_inputids += self.config.treewidth 
                    # record the cumulative probability 
                    cumulativeprob += topk 
                    # initialize the attention mask shape 
                    # TODO: prepare for the attention mask in four-dimensional 
                    attentionmaskgrowing[:, :, :, length_inputids - self.config.treewidth : length_inputids] = initialmaskblo 
                    executinginputidssparsegfou.copy_(topkidx) 
                    executingpositionidssparsee.fill_(position_marker) 
                    executingcachepositionsparsegfou.copy_(cachepositionaddergfou + length_inputids - self.config.treewidth) 
                    
                    executinginputidsfull[:, attentionmaskidx : attentionmaskidx + self.config.treewidth] = topkidx 
                    executingpositionidsfull[:, attentionmaskidx : attentionmaskidx + self.config.treewidth] = position_marker 
                    position_marker += 1 
                    attentionmaskfullgrowing[:, :, attentionmaskidx : attentionmaskidx + self.config.treewidth, : ] = attentionmaskgrowing 
                    context_selecter[selecteridx : selecteridx + self.config.treewidth] = 0 
                    selecteridx += self.config.treewidth 
                    attentionmaskidx += self.config.treewidth 
                else: 
                    self.set_inference_mode("partial") 
                    
                    gfou.replay() 
                    next_token_probs = torch.log_softmax(outputsgfou/0.6, dim = -1) # (batch_size, treewidth, vocab_size) 
                    topk, topkidx = torch.topk(next_token_probs, self.config.treewidth, dim = -1) # (batch_size, treewidth, treewidth) 
                    topknextids = topkidx.view(-1, self.config.treewidth * self.config.treewidth) 
                                
                    # filter 
                    topk = topk + cumulativeprob.unsqueeze(-1) # (batch_size, treewidth, treewidth) 
                    topk = topk.view(-1, self.config.treewidth * self.config.treewidth) # (batch_size, treewidth * treewidth) 
                    topk, topkparentidx = torch.topk(topk, k = self.config.treewidth, dim = -1) # (batch_size, treewidth) 
                    # using the parent index to get the next tokens 
                    topkparentidx = topkparentidx.squeeze(0) 
                    next_tokensidx = topknextids[:, topkparentidx] # (batch_size, treewidth) 
                    print(colored(self.tokenizer.decode(next_tokensidx[0]), "yellow"), flush = True) 
                    newpretext = [] 
                    for i in range(self.config.treewidth): 
                        oldindex = topkparentidx[i] // self.config.treewidth 
                        newpretext.append(presenttext[oldindex] + self.tokenizer.decode(next_tokensidx[0, i])) 
                    presenttext = newpretext 
                    print(presenttext) 
                    staticinputids[:, length_inputids : length_inputids + self.config.treewidth] = next_tokensidx 
                    executinginputidssparsegfou.copy_(next_tokensidx) 
                    length_inputids += self.config.treewidth 
                    topkparentidx = (topkparentidx // self.config.treewidth).squeeze(0) # (batch_size, treewidth) 
                    cumulativeprob.copy_(topk) 
                    
                    # prepare attention mask, cache position for next iteration 
                    # TODO: prepare for the attention mask in four-dimensional 
                    newattentionmask = attentionmaskgrowing.index_select(2, topkparentidx) 
                    attentionmaskgrowing.copy_(newattentionmask) 
                    
                    attentionmaskgrowing[:, :, :, length_inputids - self.config.treewidth : length_inputids] = initialmaskblo 
                    executingpositionidssparsee.fill_(position_marker) 
                    executingcachepositionsparsegfou.copy_(cachepositionaddergfou + length_inputids - self.config.treewidth) 
                    
                    executinginputidsfull[:, attentionmaskidx : attentionmaskidx + self.config.treewidth] = next_tokensidx 
                    executingpositionidsfull[:, attentionmaskidx : attentionmaskidx + self.config.treewidth] = position_marker 
                    position_marker += 1 
                    attentionmaskfullgrowing[:, :, attentionmaskidx : attentionmaskidx + self.config.treewidth, : ] = attentionmaskgrowing 
                    context_selecter[selecteridx : selecteridx + self.config.treewidth] = topkparentidx.squeeze(0) + attentionmaskidx - self.config.treewidth 
                    selecteridx += self.config.treewidth 
                    
                    attentionmaskidx += self.config.treewidth 
            else: 
                checking = True 
                executingcachepositionfull.copy_(cache_positionadder + last_check) 
                self.set_inference_mode("full") 
                print(colored(self.tokenizer.decode(executinginputidsfull[0]), "magenta"), flush = True) 
                gfull.replay() 
                
                sparse_predicted_tokens = staticinputids[:, length_inputids - checkl + 1 : length_inputids] 
                full_predicted_likelihood = torch.softmax(outputsfull/0.6, dim = -1) 
                full_predicted_likelihood2 = full_predicted_likelihood[:, context_selecter, :] 
                
                selected_likelihood = torch.gather(full_predicted_likelihood2, dim = -1, index = sparse_predicted_tokens.unsqueeze(-1)).squeeze(-1).squeeze(0) 
                
                thresholdgreatr = selected_likelihood >= self.config.thr 
                
                attentionmasksegment = attentionmaskfullgrowing[:, :, attentionmaskidx - self.config.treewidth : attentionmaskidx, last_check + 1 : last_check + checkl].squeeze(0).squeeze(0) 
                attentionmasksegment = attentionmasksegment == 0 
                
                gatherthresh = torch.masked_select(thresholdgreatr, attentionmasksegment).view(-1, kernel_size - 1) 
                lengthaccepts = torch.cumprod(gatherthresh, dim = -1, dtype = torch.long).sum(dim = -1).view(-1) 
                print("lengthaccepts {}".format(lengthaccepts), flush = True) 
                idx = torch.argmax(lengthaccepts, dim = -1) 
                print("index selected {}".format(idx), flush = True) 
                lengthaccepts = lengthaccepts[idx].item() 
                
                self.num_steps += 1 
                self.total_steps += lengthaccepts + 1 
                
                step = kernel_size - (lengthaccepts + 1) 
                executingcachepositionsparse[0] = length_inputids - checkl + (lengthaccepts + 1) 
                # prepare attention mask reset 
                attention_mask[:, length_inputids - checkl : length_inputids - checkl + lengthaccepts + 1] = 1 
                attention_mask[:, length_inputids - checkl + lengthaccepts + 1 : length_inputids] = 0 
                attentionmaskgrowing[:, :, :, length_inputids - checkl : length_inputids - checkl + lengthaccepts + 1] = 0 
                attentionmaskgrowing[:, :, :, length_inputids - checkl + lengthaccepts + 1 : length_inputids] = min_dtype 
                attentionmaskfullgrowing[:, :, :, length_inputids - checkl : length_inputids - checkl + lengthaccepts + 1] = 0 
                attentionmaskfullgrowing[:, :, :, length_inputids - checkl + lengthaccepts + 1 : length_inputids] = min_dtype 
                
                # kv cache 
                attentionmasksegment = torch.nonzero(attentionmasksegment[idx], as_tuple = True)[0] 
                if lengthaccepts == kernel_size - 1: 
                    outputsfullback = outputsfull[:, -self.config.treewidth + idx, :] 
                else: 
                    outputsfullback = outputsfull[:, attentionmasksegment[lengthaccepts], :] 
                    
                attentionmasksegment += length_inputids - checkl + 1 
                past_key_values.backgather(attentionmasksegment, length_inputids - checkl + 1, length_inputids - checkl + kernel_size) 
                
                # input ids reset 
                staticinputids[:, length_inputids - checkl + 1 : length_inputids - checkl + kernel_size] = staticinputids[:, attentionmasksegment] 
                length_inputids = length_inputids - checkl + lengthaccepts + 1 # length_inputids = length_inputids - checkl + kernel_size - step 
                last_check = length_inputids 
                next_token = torch.argmax(outputsfullback, dim = -1) 
                print(colored(self.tokenizer.decode(next_token), "blue"), flush = True) 
                presenttext = [self.tokenizer.decode(next_token) for _ in range(self.config.treewidth)] 
                print(presenttext) 
                staticinputids[:, length_inputids] = next_token 
                length_inputids += 1 
                executinginputidssparse.copy_(next_token.unsqueeze(0)) 
                executinginputidsfull[:, 0] = next_token 
                executingpositionidsfull[:, 0] = length_inputids - 1 
                
                attention_mask[:, length_inputids - 1] = 1 
                attentionmaskgrowing[:, :, :, length_inputids - 1] = 0 
                attentionmaskfullgrowing[:, :, 0, length_inputids - 1] = 0 
                # other 
                attentionmaskidx = 1 
                context_selecter.fill_(0) 
                selecteridx = 0 
                position_marker = length_inputids 
                # TODO: adding more statements here 
                cumulativeprob.fill_(0) 
                if checking and length_inputids - initial_len > 48: 
                    break 
            
        torch.cuda.synchronize() 
        endtime = time() 
        print("acceptance length on average {}".format(self.total_steps/self.num_steps)) 
        print("time taken for a single forward pass {}".format((endtime - starttime))) 
        print("inputidsselected {}".format(self.tokenizer.decode(staticinputids[0, initial_len : length_inputids])), flush = True) 
        self.generationtime += (endtime - starttime) 
        
        # Calculate elapsed time
        if self.config.griffin: 
            self.reset_states() 
        input_ids = staticinputids[:, : length_inputids] 
        
        self.totalgenerationlength += (length_inputids - initial_len - 1) # first token is not measured in range 
        print("per token generation time {}".format(self.generationtime/self.totalgenerationlength)) 

        return input_ids 
    