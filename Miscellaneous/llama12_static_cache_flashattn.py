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
from cache import StaticCache 
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

from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa 

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
        # self.start_event = torch.cuda.Event(enable_timing = True) 
        # self.end_event = torch.cuda.Event(enable_timing = True) 
    
    def forward(self, x): 
        # torch.cuda.nvtx.range_push("MLP single layer") 
        if self.profilingid == 0: 
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)) 
        else: 
            for i in range(100): # Warmup 
                # int_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x) 
                # down_proj = self.down_proj(int_states) 
                down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)) 
            torch.cuda.synchronize() 
            starttime = time() 
            for i in range(1): 
                down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)) 
            torch.cuda.synchronize() 
            endtime = time() 
            print("Time taken: {}".format(endtime - starttime)) 
            exit(0) 
        # torch.cuda.nvtx.range_pop() 
        return down_proj 

def get_llama_griffin2(model,  k_schedule):
    config = model.config
    for i, l in enumerate(model.model.layers):
        # new_mlp = GriffinLlamaMLP2(config, k_schedule[i]) 
        new_mlp = LlamaMLPnotnorma(config, k_schedule).to(model.device).to(torch.bfloat16) 
        # new_mlp.gate_proj = l.mlp.gate_proj 
        # new_mlp.up_proj = l.mlp.up_proj 
        # new_mlp.down_proj = l.mlp.down_proj 
        # new_mlp.act_fn = l.mlp.act_fn 

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
        # new_mlp = GriffinLlamaMLP2(config, k_schedule[i]) 
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

                    # GRIFFIN Expert Selection
                    # if self.config.selection_method != 'magnitude' and k_factor > 0.0: ###
                    #     k = int(int_states.shape[-1] * k_factor)
                        
                    #     neuron_stat = ((int_states / int_states.norm(dim=-1, keepdim=True))) # B, D
                    #     neuron_stat = neuron_stat.norm(dim=1)
                    #     if self.neuron_stat is None:
                    #         self.neuron_stat = neuron_stat
                    #         stat = self.neuron_stat
                    #     else:
                    #         #self.neuron_stat = self.neuron_stat
                    #         stat = (8 * neuron_stat.square() + self.neuron_stat.square()).sqrt()
                    #         self.neuron_stat = (neuron_stat.square() + self.neuron_stat.square()).sqrt()
                            
                    #     topk_weight, topk_indices = select_neurons(stat, self.config.selection_method, k)
                        
                        
                        
                    #     self.prepare_reduced_weights(topk_indices)
                        
                    down_proj = self.down_proj(int_states)

                else:
                    if k_factor == 0.0:
                        down_proj = 0 * x 
                    else:
                        #down_proj =self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

                        int_states :torch.Tensor = self.act_fn(self.gate_proj(x))

                        _, indices = torch.abs(int_states).topk(k=int(self.k_factor * int_states.shape[-1]), largest=False, dim=-1)
                        # int_states[:,:,indices] = 0 
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
                    if self.config.selection_method != 'magnitude' and k_factor > 0.0: ###
                        k = int(int_states.shape[-1] * k_factor)
                        
                        states = int_states[:,self.reference:,:]
                        
                        neuron_stat = ((states / states.norm(dim=-1, keepdim=True))) # B, D
                        neuron_stat = neuron_stat.norm(dim=1)
                        if self.neuron_stat is None:
                            self.neuron_stat = neuron_stat
                            stat = self.neuron_stat
                            
                        else:
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
                        #down_proj =self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

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
        '''
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size//2, bias = False) 
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size//2, bias = False) 
        self.down_proj = nn.Linear(self.intermediate_size//2, self.hidden_size, bias = False) 
        ''' 
        self.act_fn = F.silu 

    def forward(self, x): 
        # print("gateweight dimension {}".format(self.gate_proj.weight.data.shape)) 
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

        
        # print(colored("key_states shape {}".format(key_states.shape), "blue")) 
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
            

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        #attn_weights[1:] = nn.functional.dropout(attn_weights[1:], p=0.01)
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

class LlamaFlashAttention2(LlamaAttention):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        '''
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )
        ''' 
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2) # bsz, num_heads, q_len, head_dim 
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) # bsz, num_heads, q_len, head_dim 
        # value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) # bsz, num_heads, q_len, head_dim 
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim) 

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin) 
        
        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2) 

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype 

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        ) 
        # attn_output = query_states 

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        ) 
    
class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        # self.self_attn = (
        #     LlamaAttention(config=config, layer_idx=layer_idx)
        # ) 
        self.self_attn = (
            LlamaFlashAttention2(config = config, layer_idx = layer_idx) 
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
            attention_mask=None, 
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
        past_key_values: Optional[StaticCache] = None, 
        use_cache: bool = None,
        cache_position: Optional[bool] = None, 
    ): 
    
        batch_size, seq_length = input_ids.shape[:2]
        past_seen_tokens = 0 
        if use_cache: 
            if past_key_values is None: 
                past_key_values = StaticCache(config = self.config, max_batch_size = 1, max_cache_len = 2048, device = input_ids.device, dtype = torch.bfloat16) 
                past_seen_tokens = past_key_values.get_seq_length() 
            else: 
                past_seen_tokens = past_key_values.get_seq_length() 

        inputs_embeds = self.embed_tokens(input_ids) 
        
        if cache_position is None:
            if isinstance(past_key_values, StaticCache):
                raise ValueError("cache_position is a required argument when using StaticCache.")
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        
        if position_ids is None:
            # for verification
            position_ids = cache_position.unsqueeze(0) 
        hidden_states = inputs_embeds 
        
        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position) # for flash_attention_2, this line should be one or two if statements 

        # print("number of layers {}".format(len(self.layers))) 
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

    # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
    # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
    # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
    # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114
    def _update_causal_mask(self, attention_mask, input_tensor, cache_position):
        if self.config._attn_implementation == "flash_attention_2":
            return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if hasattr(getattr(self.layers[0], "self_attn", {}), "past_key_value"):  # static cache
            target_length = self.config.max_position_embeddings
        else:  # dynamic cache
            target_length = (
                attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor) else cache_position[-1] + 1
            )
        
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
                causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)
            elif attention_mask.dim() == 4:
                # backwards compatibility: we allow passing a 4D attention mask shorter than the input length with
                # cache. In that case, the 4D attention mask attends to the newest tokens only.
                if attention_mask.shape[-2] < cache_position[0] + sequence_length:
                    offset = cache_position[0]
                else:
                    offset = 0
                mask_shape = attention_mask.shape
                mask_slice = (attention_mask.eq(0.0)).to(dtype=dtype) * min_dtype
                causal_mask[
                    : mask_shape[0], : mask_shape[1], offset : mask_shape[2] + offset, : mask_shape[3]
                ] = mask_slice

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
        ):
            # TODO: For dynamo, rather use a check on fullgraph=True once this is possible (https://github.com/pytorch/pytorch/pull/120400).
            is_tracing = (
                torch.jit.is_tracing()
                or isinstance(input_tensor, torch.fx.Proxy)
                or (hasattr(torch, "_dynamo") and torch._dynamo.is_compiling())
            )
            if not is_tracing and torch.any(attention_mask != 1):
                # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
                # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
                # Details: https://github.com/pytorch/pytorch/issues/110213
                # causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype) 
                causal_mask = causal_mask.mul(~torch.all(causal_mask == causal_mask.min(), dim=-1)[..., None]).to(
                    dtype
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
        from transformers import AutoTokenizer 
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B") 

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


        return logits 
        
    
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs 
    ): 
        # print("attention mask pass in shape {}".format(attention_mask.shape)) 
        # this following part is for controlling the length of the input_ids 
        past_length = 0 
        cache_length = past_key_values.get_seq_length() 
        past_length = cache_length 
        max_cache_length = 2048 
        print("input_ids shape {}".format(input_ids.shape)) 
        print("past_length {}".format(past_length)) 

        if input_ids.shape[1] - past_length > 1: 
            model_inputs = {"input_ids": input_ids[:, past_length : ]} 
        else: 
            print(kwargs["executing_inputids"].shape) 
            print("input_ids[:, past_length : ] shape {}".format(input_ids[:, past_length : ].shape)) 
            kwargs["executing_inputids"].copy_(input_ids[:, past_length : ]) 
            model_inputs = {"input_ids": kwargs["executing_inputids"]} 
        
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None: # this part is included in the original prepare_inputs_for_generation 
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                # position_ids = position_ids[:, -input_ids.shape[1] :] 
                position_ids = position_ids[:, -kwargs["executing_inputids"].shape[1] :] 
        
        attention_mask = None 
        
        # preparing cache_position 
        cache_position = kwargs.get("cache_position", None) 
        if cache_position is None: 
            cache_position = torch.arange(past_length, past_length + position_ids.shape[1], device = input_ids.device, dtype = position_ids.dtype) 
            # cache_position = position_ids.clone() 
        
        position_ids = None 
        
        # print("attention_mask shape {}".format(attention_mask.shape)) 
        
        model_inputs.update(
            {
                "position_ids": position_ids, 
                "attention_mask": attention_mask, 
                "cache_position": cache_position, 
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
    '''
    def _update_model_kwargs_for_generation(
        self, 
        outputs, 
        model_kwargs, 
        num_new_tokens, 
    ): 
        if (
            model_kwargs.get("use_cache", True) 
            and "cache_position" in model_kwargs 
            and model_kwargs["cache_position"] is not None 
        ): 
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens 
        
        return model_kwargs 
    ''' 
    def _update_model_kwargs_for_generation(
        self,
        outputs, 
        model_kwargs, 
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
        num_new_tokens: int = 1,
    ): 
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state 

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                    dim=-1,
                )

        if (
            model_kwargs.get("use_cache", True)
            and "cache_position" in model_kwargs
            and model_kwargs["cache_position"] is not None
        ):
            model_kwargs["cache_position"].copy_(model_kwargs["cache_position"][-1:] + num_new_tokens) 

        return model_kwargs
    
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
        r"""
        Generates sequences of token ids for models with a language modeling head using **greedy decoding** and can be
        used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin.greedy_search`] directly. Use generate()
        instead. For an overview of generation strategies and code examples, check the [following
        guide](../generation_strategies).

        </Tip>


        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.

            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            output_logits (`bool`, *optional*, defaults to `False`):
                Whether or not to return the raw prediction logit scores. See `logits` under returned tensors
                for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
                If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForCausalLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     StoppingCriteriaList,
        ...     MaxLengthCriteria,
        ... )

        >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

        >>> # set pad_token_id to eos_token_id because GPT2 does not have a PAD token
        >>> model.generation_config.pad_token_id = model.generation_config.eos_token_id

        >>> input_prompt = "It might be possible to"
        >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(10, eos_token_id=model.generation_config.eos_token_id),
        ...     ]
        ... )
        >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

        >>> outputs = model.greedy_search(
        ...     input_ids, logits_processor=logits_processor, stopping_criteria=stopping_criteria
        ... )

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ["It might be possible to get a better understanding of the nature of the problem, but it's not"]
        ```"""
        initial_len = input_ids.shape[1] 
        '''
        if input_ids.shape[1] > 50: 
            input_ids = input_ids[:, : 50 - 1] 
            if model_kwargs["attention_mask"] is not None: 
                model_kwargs["attention_mask"] = model_kwargs["attention_mask"][:, : 50 - 1] 
            # if model_kwargs["position_ids"] is not None: 
                # model_kwargs["position_ids"] = model_kwargs["position_ids"][:, : 99] 
        ''' 
        kernel_size = self.config.kernel_size
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList() 
        '''
        for stoppingcrition in stopping_criteria: 
            if isinstance(stoppingcrition, MultiTokenEOSCriteria): 
                stoppingcrition.sequence_id_len = len(stoppingcrition.sequence_ids) + self.config.kernel_size 
        ''' 
        if max_length is not None:
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        ) 
        
        # return_dict_in_generate = True 

        # init attention / hidden states / scores tuples
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None
        
        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )
        
        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False  # used by synced_gpus only
        initial_len = input_ids.shape[1] # prefill length 
        last_check = initial_len
        track_position_ids = None
        track_cache_position = None
        want_to_quit = False
        approve_quit = False 
        last_input_ids_print_pos = initial_len # this variable is for visualization 
        outputs = None 
        past_key_values = StaticCache(config = self.config, max_batch_size = 1, max_cache_len = 2048, device = input_ids.device, dtype = torch.bfloat16) 
        model_kwargs["past_key_values"] = past_key_values 
        model_kwargs["executing_inputids"] = torch.empty((1, 1)).to(input_ids.device, torch.long) 
        
        assert input_ids.shape[0] == 1 
        # Start event
        # End event
        timecollection = [] 
        
        # TODO the main while loop 
        while input_ids.shape[1] - initial_len < 50: 
            
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs) 
            # print("{} ".format(input_ids.shape[1]), flush = True, end = " ") 
            
            track_position_ids = model_inputs["position_ids"] if track_position_ids is None else torch.cat([track_position_ids, model_inputs["position_ids"]], dim = -1) 
            track_cache_position = model_inputs["cache_position"] if track_cache_position is None else torch.cat([track_cache_position, model_inputs["cache_position"]], dim = -1) 
            
            currentlength = input_ids.shape[1] # accumulating new_tokens 
            generatedlength = currentlength - initial_len 
            
            # past_key_values.mode = "decoding" 
            
            if generatedlength > 0: # using partial sparse model 
            # if generatedlength == 0: 
                # with torch.autograd.profiler.emit_nvtx():
                # torch.cuda.nvtx.range_push("profiling one iteration") 
                past_key_values = model_inputs["past_key_values"] 
                # past_key_values.mode = "decoding" 
                # input_embeds = torch.randn(1, 1000, 4096).cuda().to(torch.bfloat16) 
                # model_inputs["input_ids"] = model_inputs["input_ids"][:, -1].unsqueeze(1) 
                # model_inputs["attention_mask"] = None 
                # model_inputs["position_ids"] = None 
                print("input shape {}".format(model_inputs["input_ids"].shape)) 
                outputs = torch.empty((1, 1, self.config.vocab_size), device = input_ids.device, dtype = torch.bfloat16) 
                
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                for i in range(100): 
                    outputs = self( 
                        input_ids = model_inputs["input_ids"], 
                        attention_mask = model_inputs["attention_mask"], 
                        position_ids = model_inputs["position_ids"], 
                        past_key_values = model_inputs["past_key_values"], 
                        cache_position = model_inputs["cache_position"], 
                        # inputs_embeds = input_embeds, 
                    ) 
                    # length = past_key_values.get_seq_length() 
                    past_key_values.rollbackone() 
                    # lengthtwo = past_key_values.get_seq_length() 
                    # assert lengthtwo == length - 1, "rollback is not" 
                torch.cuda.current_stream().wait_stream(s)
                print("warming up") 
                
                g = torch.cuda.CUDAGraph() 
                
                with torch.cuda.graph(g): 
                    outputs = self( 
                        input_ids = model_inputs["input_ids"], 
                        attention_mask = model_inputs["attention_mask"], 
                        position_ids = model_inputs["position_ids"], 
                        use_cache = True, 
                        past_key_values = model_inputs["past_key_values"], 
                        cache_position = model_inputs["cache_position"], 
                        # inputs_embeds = input_embeds, 
                    ) 
                    past_key_values.rollbackone() 
                
                torch.cuda.synchronize() 
                starttime = time() 
                for i in range(1000): 
                    '''
                    outputs = self(
                        input_ids = model_inputs["input_ids"], 
                        # inputs_embeds = input_embeds, 
                        attention_mask = model_inputs["attention_mask"], 
                        position_ids = model_inputs["position_ids"], 
                        use_cache = model_inputs["use_cache"], 
                        past_key_values = model_inputs["past_key_values"], 
                        cache_position = model_inputs["cache_position"], 
                    ) 
                    ''' 
                    g.replay() 
                    past_key_values.rollbackone() 
                torch.cuda.synchronize() 
                endtime = time() 
                print("time taken for a single forward pass {}".format((endtime - starttime)/1000)) 
                # torch.cuda.nvtx.range_pop() 
                
                next_token_logits = outputs[:, -1, :] 
                next_tokens_scores = logits_processor(input_ids, next_token_logits) 
                next_token = torch.argmax(next_tokens_scores, dim = -1) 
                print("next token {}".format(self.tokenizer.decode(next_token)), flush = True) 
                
                for i in range(20): 
                    input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim = -1) 
                    # input_ids.copy_(torch.cat([input_ids, next_token.unsqueeze(0)], dim = -1)) 
                    model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder = False) 
                    print("model_kwargs past_key_values None {}".format(model_kwargs["past_key_values"] == None)) 
                    model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs) 
                    
                    g.replay() 
                    
                    next_token_logits = outputs[:, -1, :] 
                    next_tokens_scores = logits_processor(input_ids, next_token_logits) 
                    next_token = torch.argmax(next_tokens_scores, dim = -1) 
                    print("next token {}".format(self.tokenizer.decode(next_token)), flush = True) 
                
                exit(0) 
                
            else: # using the full model 
                outputs = self(
                    input_ids = model_inputs["input_ids"], 
                    attention_mask = None, 
                    position_ids = None, 
                    use_cache = True, 
                    past_key_values = model_inputs["past_key_values"], 
                    cache_position = model_inputs["cache_position"], 
                ) 
            
            # depending on checking, whether we need to correct or roll back outputs 
            check_flag = False 
            if self.config.check: 
                if ((currentlength - last_check) == kernel_size and generatedlength > 0) or want_to_quit: 
                    # getting all the input to the full model 
                    check_flag = True 
                    past_key_values = outputs.past_key_values 
                    past_key_values.mode = "checking" # cache is being roll back 
                    checklength = currentlength - last_check 
                    check_input_ids = input_ids[:, -checklength:] 
                    check_attention_mask = model_inputs["attention_mask"] 
                    check_position_ids = track_position_ids[:, -checklength:] 
                    check_cache_position = track_cache_position[-checklength :] 
                    
                    self.set_inference_mode("full") 
                    # calling the evaluation 
                    outputs = self(
                        input_ids = check_input_ids, 
                        attention_mask = check_attention_mask, 
                        position_ids = check_position_ids, 
                        past_key_values = past_key_values, 
                        use_cache = True, 
                        cache_position = check_cache_position, 
                        return_dict = True, 
                        output_attentions = output_attentions, 
                        output_hidden_states = output_hidden_states, 
                    ) 
                    sparse_predicted_tokens = check_input_ids[:, 1:] 
                    full_predicted_logits = outputs.logits[:, :-1, :] 
                    full_predicted_likelihood = torch.softmax(full_predicted_logits / 0.6, dim = -1) 
                    selected_likelihood = torch.gather(full_predicted_likelihood, dim = -1, index = sparse_predicted_tokens.unsqueeze(-1)).squeeze(-1).squeeze(0) 

                    # getting the length of the accepted token 
                    lengthaccepts = 0 
                    requirerollback = False 
                    for i in range(selected_likelihood.shape[0]): 
                        if selected_likelihood[i] >= self.config.thr: 
                            lengthaccepts += 1 
                        else: 
                            # first pass skip correct 
                            if self.config.secondrollback == True: 
                                tokencorrection = torch.argmax(full_predicted_likelihood[:, i, :], dim = -1) 
                                print("\nfirst pass lengthaccepts {} {}to{}".format(lengthaccepts, self.tokenizer.decode(sparse_predicted_tokens[0, i]), self.tokenizer.decode(tokencorrection[0])), flush = True) 
                                check_input_ids[:, i + 1] = tokencorrection 
                                input_ids[:, last_check + i + 1] = tokencorrection 
                                requirerollback = True 
                            break 
                    
                    if self.config.secondrollback == True and requirerollback: 
                        outputs = self(
                            input_ids = check_input_ids, 
                            attention_mask = check_attention_mask, 
                            position_ids = check_position_ids, 
                            past_key_values = past_key_values, 
                            use_cache = True, 
                            cache_position = check_cache_position, 
                            return_dict = True, 
                            output_attentions = output_attentions, 
                            output_hidden_states = output_hidden_states, 
                        ) 
                        
                        sparse_predicted_tokens = check_input_ids[:, 1:] 
                        full_predicted_logits = outputs.logits[:, :-1, :] 
                        full_predicted_likelihood = torch.softmax(full_predicted_logits / 0.6, dim = -1) 
                        selected_likelihood = torch.gather(full_predicted_likelihood, dim = -1, index = sparse_predicted_tokens.unsqueeze(-1)).squeeze(-1).squeeze(0) 

                        # getting the length of the accepted token 
                        lengthaccepts = 0 
                        for i in range(selected_likelihood.shape[0]): 
                            if selected_likelihood[i] >= self.config.thr: 
                                lengthaccepts += 1 
                            else: 
                                # second pass mandatory roll back 
                                print("second pass lengthaccepts {} {}".format(lengthaccepts, self.tokenizer.decode(sparse_predicted_tokens[0, i])), flush = True) 
                                break 
                    
                    outputs.logits = outputs.logits[:, : (lengthaccepts + 1), :] 
                    # step = kernel_size - (lengthaccepts + 1) 
                    step = checklength - (lengthaccepts + 1) 
                    '''
                    if self.verbose: 
                        print(colored(self.tokenizer.decode(input_ids[0, last_input_ids_print_pos : last_check + lengthaccepts + 1]), "green"), flush = True, end = " ") 
                        if step > 0: 
                            for i in range(step): 
                                print(colored(self.tokenizer.decode(input_ids[0, -step + i]), "red"), flush = True, end = "") 
                                print(colored("({:.2f})".format(selected_likelihood[lengthaccepts + i]), "red"), flush = True, end = " ") 
                        # print(colored(self.tokenizer.decode(input_ids[0, last_input_ids_print_pos + last_satisfy :]), "red"), flush = True, end = " ") 
                    ''' 
                    # roll back 
                    past_key_values.rollback(step) 
                    past_key_values.mode = "decoding" 
                    last_input_ids_print_pos = last_check + lengthaccepts + 1 
                    if step > 0: 
                        input_ids = input_ids[:, :-step] 
                        track_position_ids = track_position_ids[:, :-step] 
                        track_cache_position = track_cache_position[:-step] 
                    
                    # setup for next iteration 
                    last_check = input_ids.shape[1] 
                    self.num_steps += 1 
                    self.total_steps += lengthaccepts + 1 
                    if lengthaccepts != checklength - 1: 
                        self.errorinstance += 1 
                        self.total_roll_back_length_error += (checklength - 1 - lengthaccepts) 
                        self.roll_back_length_in_error.append(checklength - 1 - lengthaccepts) 
                    if step == 0: # although not needed every time, good to have here 
                        approve_quit = True 
                
            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs[:, -1, :] 

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits) 

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)
            
            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1) 
            print("input_ids shape {}".format(input_ids.shape)) 
            
            if self.verbose and not self.config.check: 
                print(self.tokenizer.decode(input_ids[0, -1]), flush = True, end = " ") 
            '''
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, num_new_tokens = 1, 
            ) 
            ''' 
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder 
            ) 
            
            if check_flag: 
                model_kwargs["attention_mask"] = model_kwargs["attention_mask"][...,:input_ids.shape[1]] 
            
            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None: 
                print("next token shape {} eos_token_id_tensor shape {}".format(next_tokens.shape, eos_token_id_tensor.shape), flush = True) 
                if torch.eq(next_tokens.expand(eos_token_id_tensor.shape[0]), eos_token_id_tensor).any(): 
                    this_peer_finished = True 

            # stop if we exceed the maximum length
            if stopping_criteria(input_ids, scores):
                this_peer_finished = True 
                '''
                this_peer_finished = this_peer_finished and check_flag 
                
                if this_peer_finished: 
                    recheckoutcome = False 
                    
                    from transformers.generation.stopping_criteria import MaxLengthCriteria 
                    for stoppingc in stopping_criteria: 
                        if isinstance(stoppingc, MaxLengthCriteria): 
                            recheckoutcome = recheckoutcome or stoppingc(input_ids, scores) 
                            
                        if isinstance(stoppingc, MultiTokenEOSCriteria): 
                            stoppingc.done_tracker = [False] * input_ids.shape[0] 
                            # print("stop sequence {}".format(stoppingc.sequence)) 
                            # stoppingtest = MultiTokenEOSCriteria2(stoppingc.sequence, self.tokenizer, stoppingc.initial_decoder_input_length, 1) 
                            # print("meet the multi token eos criteria {}".format(stoppingtest(input_ids, scores))) 
                            recheckoutcome = recheckoutcome or stoppingc(input_ids, scores) 
                            # print("meet the multi token eos criteria {}".format(stoppingc(input_ids, None))) 
                            # print("meet the multi token eos criteria {}".format(stoppingc(input_ids, scores))) 
                    this_peer_finished = recheckoutcome 
                ''' 
            else: 
                this_peer_finished = False 

            # print("stopping_criteria {}".format(stopping_criteria)) 
                                
            if this_peer_finished and not synced_gpus: 
                want_to_quit = True 
                if approve_quit or (not self.config.check): 
                    break 
        
        # Calculate elapsed time
        
        print("average time: {}".format(np.average(timecollection)), flush = True) 
        print(timecollection, flush = True) 
        
        if self.config.griffin: 
            self.reset_states() 
        track_position_ids = None 
        track_cache_position = None 
        
        exit(0) 
        
        self.totalgenerationlength += input_ids.shape[1] - initial_len 

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else: 
            return input_ids
        