# Adapted from Hugging Face implementation

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.llama.modeling_llama import (
LlamaConfig, LlamaRotaryEmbedding, LlamaDynamicNTKScalingRotaryEmbedding, LlamaLinearScalingRotaryEmbedding,
LlamaRMSNorm, LlamaPreTrainedModel, LlamaMLP,
apply_rotary_pos_emb, repeat_kv
)
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
import math
import warnings
from typing import List, Optional, Tuple, Union
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import torch.nn as nn
import torch
from transformers.cache_utils import Cache, DynamicCache, StaticCache, SinkCache
import torch.nn.functional as F
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast
)
from torch.nn import CrossEntropyLoss

from transformers.generation.beam_constraints import DisjunctiveConstraint, PhrasalConstraint
from transformers.generation.beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
from transformers.generation.candidate_generator import (
    AssistedCandidateGenerator,
    CandidateGenerator,
    PromptLookupCandidateGenerator,
    _crop_past_key_values,
    _prepare_attention_mask,
    _prepare_token_type_ids,
)
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    EncoderRepetitionPenaltyLogitsProcessor,
    EpsilonLogitsWarper,
    EtaLogitsWarper,
    ExponentialDecayLengthPenalty,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    ForceTokensLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitNormalization,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    MinNewTokensLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SequenceBiasLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
    UnbatchedClassifierFreeGuidanceLogitsProcessor,
)
from transformers.generation.stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)

from cache import GriffinCache
if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.generation.streamers import BaseStreamer

from termcolor import colored 

import numpy as np 

from lm_eval.models.utils import MultiTokenEOSCriteria 


NEED_SETUP_CACHE_CLASSES_MAPPING = {
    "static": StaticCache,
}

from transformers.generation.utils import GenerateOutput, GenerateDecoderOnlyOutput, GenerateNonBeamOutput, GenerateEncoderDecoderOutput
import time 
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
def get_llama_griffin(model,  k_schedule, patternstrict): 
    config = model.config
    for i, l in enumerate(model.model.layers):
        new_mlp = GriffinLlamaMLP(config, k_schedule[i], patternstrict) 
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

def get_llama_griffin2(model,  k_schedule):
    config = model.config
    for i, l in enumerate(model.model.layers):
        new_mlp = GriffinLlamaMLP2(config, k_schedule[i])
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
    def __init__(self, config, k_factor, patternstrict): 
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
        self.patternstrict = patternstrict 
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

                        int_states :torch.Tensor = self.act_fn(self.gate_proj(x)) # B, seq_len, D 

                        if not self.patternstrict: 
                            assert int_states.shape[1] == 1 
                            _, indices = torch.abs(int_states).topk(k=int(self.k_factor * int_states.shape[-1]), largest=False, dim=-1) # B, seq_len, k 
                            # int_states[:,:,indices] = 0 
                            buffrzeros = torch.zeros_like(int_states) 
                            int_states.scatter_(dim = -1, index = indices, src = buffrzeros) 
                        else: 
                            
                            batch_size, seq_len, hidden_size = int_states.shape 
                            intstatesbuffer = int_states.permute(1, 0, 2) # seq_len, batch_size, hidden_size 
                            intstatesbuffer = intstatesbuffer.abs().sum(dim = 1) # seq_len, hidden_size 
                            # intstatesbuffer = intstatesbuffer/intstatesbuffer.norm(dim = -1, keepdim = True) 
                            # intstatesbuffer = intstatesbuffer.norm(dim = 1) # seq_len, hidden_size 
                            _, indices = intstatesbuffer.topk(k = int(self.k_factor * hidden_size), largest = False, dim = -1) 
                            indices = indices.unsqueeze(0).expand(batch_size, -1, -1) # B, seq_len, k 
                            buffrzeros = torch.zeros_like(int_states) 
                            int_states.scatter_(dim = -1, index = indices, src = buffrzeros) 
                            
                            '''
                            batch_size, seq_len, hidden_size = int_states.shape 
                            assert seq_len == 1 
                            intstatesbuffer = int_states.permute(1, 0, 2) # seq_len, B, hidden_size 
                            _, indices = torch.abs(intstatesbuffer).topk(k = int(self.k_factor * hidden_size), largest = False, dim = -1) # indices is of shape (seq_len, B, k) 
                            indices = indices.reshape(seq_len, (batch_size * int(self.k_factor * hidden_size))) # seq_len, B * hidden_size 
                            indices = indices.unique(dim = -1) # seq_len, m m larger than or equal to k 
                            selectedintstates = intstatesbuffer.gather(dim = -1, index = indices.unsqueeze(1).expand(-1, batch_size, -1)) # seq_len, B, m 
                            selectedintstates = selectedintstates.abs().sum(dim = 1) # seq_len, m 
                            _, final_indices = selectedintstates.topk(k = int(self.k_factor * hidden_size), largest = False, dim = -1) 
                            # _, final_indices = intstatesbuffer.abs().sum(dim = 1).topk(k = int(self.k_factor * hidden_size), largest = False, dim = -1) 
                            final_indices = final_indices.unsqueeze(0).expand(batch_size, -1, -1) # seq_len, B, k 
                            # final_indices = final_indices.permute(1, 0, 2) # B, seq_len, k 
                            buffrzeros = torch.zeros_like(int_states) 
                            int_states.scatter_(dim = -1, index = final_indices, src = buffrzeros) 
                            ''' 
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
        past_key_value: Optional[GriffinCache] = None,
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

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states
        
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""

class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

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

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[GriffinCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = 0
        if use_cache:  # kept for BC (cache positions)
            if not isinstance(past_key_values, StaticCache):
                if past_key_values is not None:
                    past_seen_tokens = past_key_values.get_seq_length()
                else:
                    past_key_values = GriffinCache()
                    past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            if isinstance(past_key_values, StaticCache):
                raise ValueError("cache_position is a required argument when using StaticCache.")
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position)

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache if isinstance(next_decoder_cache, Cache) else next_decoder_cache
            )
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
    # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
    # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
    # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114
    def _update_causal_mask(self, attention_mask, input_tensor, cache_position):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
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
                causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.total_steps = 0
        self.num_steps = 0
        self.num_sentence = 0 
        self.totalgenerationlength = 0 
        self.total_roll_back_length_error = 0 
        self.roll_back_length_in_error = [] 
        self.errorinstance = 0 
        self.verbose = False # manually set to false during measurement 
        self.flattentreesize = 0 
        self.batchsizecount = 0 # not for statistics presentation, but for intermediate states 
        self.averagedraftingbatchsize = 0 # for measuring the tree growing size 
        
        # for bug debugging investigation only 
        from transformers import AutoTokenizer 
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B") 
        if self.verbose: 
            assert self.tokenizer is not None 
        # Initialize weights and apply final processing
        self.post_init() 
        
        # tree related 
        self.k = 8 
        self.beamwidth = self.k 
        print("beam width is {}".format(self.beamwidth)) 
        self.beam = [] 
        self.completed_sequences = [] 
    
    def reset_tree(self): 
        self.beam = [] 
        self.completed_sequences = [] 

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def updatestatistic(self): 
        self.num_steps = 0 
        self.num_sentence = 0 
        self.total_steps = 0 
        self.totalgenerationlength = 0 
        self.total_roll_back_length_error = 0 
        self.roll_back_length_in_error = [] 
        self.errorinstance = 0 
        self.flattentreesize = 0 
        self.batchsizecount = 0 
        self.averagedraftingbatchsize = 0 

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ) 
        
    def checkcompletedsequences(self, seq): 
        for seq1, _, _ in self.completed_sequences: 
            if torch.equal(seq, seq1): 
                return True 
        return False 
    
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values = None, attention_mask=None, inputs_embeds=None, **kwargs 
    ): 
        active_seq = [] 
        active_log_prob = [] 
        active_cache = [] 
        # print("length of completed_sequences {}".format(len(self.completed_sequences))) 
        # process tree input_ids and kv cache 
        for seq, cum_log_prob, kv_cache in self.beam: 
            # print("seq[:, -1] {}".format(seq[0][-1].item())) 
            # print("self.tokenizer.eos_token_id {}".format(self.tokenizer.eos_token_id)) 
            # print("seq[:, -1].item() == self.tokenizer.eos_token_id {}".format(seq[0][-1].item() == self.tokenizer.eos_token_id)) 
            if seq[0][-1].item() == self.tokenizer.eos_token_id: 
                print(colored("adding to completed sequences", "green")) 
                exit(0) 
                if not self.checkcompletedsequences(seq): 
                    self.completed_sequences.append((seq, cum_log_prob, kv_cache)) 
            else: 
                active_seq.append(seq) 
                active_log_prob.append(cum_log_prob) 
                active_cache.append(kv_cache) 
        
        if len(active_seq) == 0: 
            return {"input_ids": None} 
        
        # stack 
        if len(active_seq) > 1: 
            # input_ids = torch.stack(active_seq, dim = 0) 
            input_ids = torch.cat(active_seq, dim = 0) 
            past_key_values = GriffinCache.stackcache(active_cache) 
        else: 
            input_ids = active_seq[0] 
            past_key_values = active_cache[0] 
        extendedinput_ids = input_ids.clone() 
        
        past_length = 0 
        assert past_key_values is not None 
        if past_key_values is not None: 
            if isinstance(past_key_values, GriffinCache): 
                cache_length = past_key_values.get_seq_length() # cache_length is not useful 
                past_length = past_key_values.seen_tokens 
                max_cache_length = None 
            else: 
                cache_length = past_length = past_key_values[0][0].shape[2] 
                max_cache_length = None 
        
            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]: 
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ): # I don't think this condition is used 
                attention_mask = attention_mask[:, -max_cache_length:]
        
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None: # this part is included in the original prepare_inputs_for_generation 
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # preparing cache_position 
        if getattr(self.model.layers[0].self_attn, "past_key_value", None) is not None: # I don't think this block of code is necessary 
            past_key_value_first_layer = self.model.layers[0].self_attn.past_key_value 
            past_length = past_key_value_first_layer.get_seq_length() 
            input_ids = input_ids[:, past_length :] 
            position_ids = position_ids[:, past_length :] 
        
        # preparing cache_position 
        cache_position = kwargs.get("cache_position", None) 
        if cache_position is None: 
            # print(colored("cache_position is None", "red")) 
            cache_position = torch.arange(past_length, past_length + position_ids.shape[1], device = input_ids.device, dtype = position_ids.dtype) 
            # cache_position = position_ids.clone() 
        
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids} 
        
        model_inputs.update(
            {
                "position_ids": position_ids, 
                "attention_mask": attention_mask, 
                "cache_position": cache_position, 
                "past_key_values": past_key_values, 
                "use_cache": kwargs.get("use_cache"), 
                "extended_input_ids": extendedinput_ids, 
                "active_probs": active_log_prob, 
            } 
        ) 
        return model_inputs 
    
    def prepare_initial_run(self, input_ids, attention_mask, inputs_embeds = None, **kwargs): 
        past_length = 0 
        
        past_key_values = None 
        # position_ids 
        position_ids = kwargs.get("position_ids", None) 
        if attention_mask is not None and position_ids is None: # this part is included in the original prepare_inputs_for_generation 
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :] 
        
        # preparing cache_position 
        if getattr(self.model.layers[0].self_attn, "past_key_value", None) is not None: # I don't think this block of code is necessary 
            past_key_value_first_layer = self.model.layers[0].self_attn.past_key_value 
            past_length = past_key_value_first_layer.get_seq_length() 
            input_ids = input_ids[:, past_length :] 
            position_ids = position_ids[:, past_length :] 
        
        # preparing cache_position 
        cache_position = kwargs.get("cache_position", None) 
        if cache_position is None: 
            cache_position = torch.arange(past_length, past_length + position_ids.shape[1], device = input_ids.device, dtype = position_ids.dtype) 
        
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        
        model_inputs.update(
            {
                "position_ids": position_ids, 
                "attention_mask": attention_mask, 
                "cache_position": cache_position, 
                "past_key_values": past_key_values, 
                "use_cache": kwargs.get("use_cache"), 
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
    
    def reset_states(self): 
        for layer in self.model.layers: 
            layer.mlp.reset_stats() 
    
    def set_inference_mode(self, mode): 
        for layer in self.model.layers: 
            layer.mlp.inference_mode = mode 
    
    def merging_tree_into_one_sequence2(self, treetensor): 
        num_sequences, sequence_length = treetensor.shape
        # treetensor = treetensor.clone().cpu() 
        merge_sequence = []
        visited = set()

        def dfs(sequence, depth=0):
            for i in range(depth, sequence_length):
                current_element = sequence[i].item()
                current_path = tuple(sequence[:i+1].cpu().tolist()) 
                
                if current_path not in visited:
                    visited.add(current_path)
                    merge_sequence.append(current_element)
                    
                    for other_sequence in treetensor:
                        if torch.equal(other_sequence[:i+1], sequence[:i+1]):
                            if i+1 < sequence_length:
                                dfs(other_sequence, i+1)
        
        # Start DFS from each sequence
        for seq in treetensor:
            dfs(seq)

        return torch.tensor(merge_sequence), len(merge_sequence)
    
    def merging_tree_into_one_sequence(self, treetensor): 
        num_sequences, sequence_length = treetensor.shape 
        merge_sequence = treetensor[0].tolist() 
        for sequence in treetensor[1:]: 
            i = 0 
            while i < sequence_length and i < len(merge_sequence) and sequence[i] == merge_sequence[i]: 
                i += 1 
            merge_sequence.extend(sequence[i:]) 
        return len(merge_sequence) 
    
    def rollbacklastchunkstatistic(self, 
                                   last_total_step, 
                                   last_error_instance, 
                                   last_roll_back_length_error, 
                                   last_flatten_tree_size, 
                                   last_average_drafting_batch_size): 
        self.num_steps -= 1 
        # num_sentence shouldn't be rolled back 
        self.total_steps -= last_total_step 
        # total generation length also shouldn't be rolled back 
        self.errorinstance -= last_error_instance 
        self.total_roll_back_length_error -= last_roll_back_length_error 
        self.flattentreesize -= last_flatten_tree_size 
        # batch size count also shouldn't be rolled back 
        self.averagedraftingbatchsize -= last_average_drafting_batch_size 
    
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
        streamer: Optional["BaseStreamer"] = None,
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
        
        kernel_size = self.config.kernel_size 
        self.k = self.config.treewidth 
        if self.beamwidth != self.k: 
            self.beamwidth = self.k 
            print("beamwidth is {}".format(self.beamwidth)) 
        self.reset_tree() 
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList() 
        for stoppingcrition in stopping_criteria: 
            if isinstance(stoppingcrition, MultiTokenEOSCriteria): 
                stoppingcrition.sequence_id_len = len(stoppingcrition.sequence_ids) + self.config.kernel_size 
        
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
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
        # approve_quit = False 
        approve_quit = True 
        last_input_ids_print_pos = initial_len # this variable is for visualization 
        outputs = None 
        
        self.num_sentence += 1 
        assert input_ids.shape[0] == 1 # the purpose of the input_ids now is to hold the collapsed tree results 
        
        # full model initial run 
        model_inputs = self.prepare_initial_run(input_ids, **model_kwargs) 
        track_position_ids = model_inputs["position_ids"] if track_position_ids is None else torch.cat([track_position_ids, model_inputs["position_ids"]], dim = -1) 
        track_cache_position = model_inputs["cache_position"] if track_cache_position is None else torch.cat([track_cache_position, model_inputs["cache_position"]], dim = -1) 
        
        currentlength = input_ids.shape[1] # accumulating new_tokens 
        
        self.set_inference_mode("full") 
        outputs = self( # full model run 
            **model_inputs, 
            return_dict = True, 
            output_attentions = output_attentions, 
            output_hidden_states = output_hidden_states, 
        ) 
        
        # if synced_gpus and this_peer_finished:
        #     continue  # don't waste resources running the code we don't need 
        
        next_token_logits = outputs.logits[:, -1, :]
        # pre-process distribution
        next_tokens_scores = logits_processor(input_ids, next_token_logits) 
        initial_log_probability = torch.log_softmax(next_tokens_scores, dim = -1) 
        # Store scores, attentions and hidden_states when required 
        
        # argmax
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)
        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        ) 
        
        if self.verbose: 
            print(colored(self.tokenizer.decode(next_tokens), color = "cyan"), flush = True) 
        self.beam = [(input_ids, initial_log_probability[0][next_tokens.item()], outputs.past_key_values)] 
        iteration_count = -1 
        
        # TODO the main while loop 
        while True: 
            iteration_count += 1 
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs) 
            all_sequences = [x for x in self.completed_sequences] # appending to all_sequences shouldn't affect the completed_sequences 
            if model_inputs["input_ids"] is None: 
                break 
            
            track_position_ids = model_inputs["position_ids"] if track_position_ids is None else torch.cat([track_position_ids, model_inputs["position_ids"]], dim = -1) 
            track_cache_position = model_inputs["cache_position"] if track_cache_position is None else torch.cat([track_cache_position, model_inputs["cache_position"]], dim = -1) 
            
            currentlength = input_ids.shape[1] # accumulating new_tokens 
            generatedlength = currentlength - initial_len 

            self.set_inference_mode("partial") 
            model_inputs["past_key_values"].mode = "decoding" 
            self.batchsizecount += model_inputs["input_ids"].shape[0] 
            if self.config.check and (currentlength - last_check) == kernel_size or (self.config.check and want_to_quit): # to avoid one more pass of the sparse model, here we use a dummy operation to fill in one step into the kv cache 
                # outputs.past_key_values.adding_one_entry() 
                # print("model_inputs[past_key_values].shape {}".format(model_inputs["past_key_values"].key_cache[0].shape)) 
                model_inputs["past_key_values"].adding_one_entry() 
                # print("adding one input_ids shape {}".format(model_inputs["extended_input_ids"].shape)) 
                # print("se")
                # print("after addingone model_inputs[past_key_values].shape {}".format(model_inputs["past_key_values"].key_cache[0].shape)) 
            else: 
                # print("input_ids shape {}".format(model_inputs["input_ids"].shape)) 
                # print("extended input_ids shape {}".format(model_inputs["extended_input_ids"].shape)) 
                # print("cache length {}".format(model_inputs["past_key_values"].get_seq_length())) 
                outputs = self( # I expand the arguments of model_inputs 
                    input_ids = model_inputs["input_ids"], 
                    position_ids = model_inputs["position_ids"], 
                    attention_mask = model_inputs["attention_mask"], 
                    past_key_values = model_inputs["past_key_values"], 
                    use_cache = model_inputs["use_cache"], 
                    cache_position = model_inputs["cache_position"], 
                    return_dict = True, 
                    output_attentions = output_attentions, 
                    output_hidden_states = output_hidden_states, 
                ) 
                
                next_token_logits = outputs.logits[:, -1, :] 
                next_tokens_scores = logits_processor(input_ids, next_token_logits)
                next_token_probs = torch.log_softmax(next_token_logits/0.6, dim = -1) # batch_size, seq_len, vocab_size 
                now_cache = outputs.past_key_values 
                active_now_cache = GriffinCache.splitcache(now_cache) 

                for i in range(next_token_logits.shape[0]): 
                    topk, topkidx = torch.topk(next_token_probs[i], self.k, dim = -1) # Decoding line here, trying to instead of finding the top1, we find topk 
                    for logprob, ids in zip(topk, topkidx): 
                        if ids.item() == self.tokenizer.eos_token_id: 
                            continue # ignore sequences that have eos_token 
                        extended_input_ids = model_inputs["extended_input_ids"][i] 
                        outputlogprob = logprob 
                        logprob = model_inputs["active_probs"][i] + logprob.item() # TODO print out logprob shape 
                        extended_input_ids = torch.cat([extended_input_ids, ids.unsqueeze(0)], dim = 0) # ids is of shape (batch_size, 1) 
                        extended_input_ids = extended_input_ids.unsqueeze(0) 
                        all_sequences.append((extended_input_ids, logprob, i)) 
                        if self.verbose: 
                            print(colored("{:.5f} * {:.5f} = {:.5f}".format(torch.exp(model_inputs["active_probs"][i]), torch.exp(outputlogprob), torch.exp(model_inputs["active_probs"][i] + logprob)), color = "light_green"), flush = True) 
                            print(colored("({}, {}, {})".format(self.tokenizer.decode(extended_input_ids[0, initial_len :]), logprob, i), color = "yellow"), flush = True) 
                        if torch.exp(outputlogprob) > 0.95 and self.config.filteractiveenabled: 
                            break 
                # prune 
                all_sequences = sorted(all_sequences, key = lambda x: x[1], reverse = True) 
                self.beam = [] 
                for i in range(min(self.beamwidth, len(all_sequences))): 
                    if isinstance(all_sequences[i][2], GriffinCache): 
                        new_cache_copy = all_sequences[i][2].copy() 
                    else: 
                        new_cache_copy = active_now_cache[all_sequences[i][2]].copy() 
                    self.beam.append((all_sequences[i][0], all_sequences[i][1], new_cache_copy)) 
                    if self.verbose: 
                        print(colored("({}, {}, {})".format(self.tokenizer.decode(self.beam[-1][0][0, initial_len :]), self.beam[-1][1], self.beam[-1][2].seen_tokens), color = "light_magenta"), flush = True) 
                if self.verbose: 
                    print() 
            
            check_flag = False 
            last_total_step = 0 
            last_roll_back_length_error = 0 
            last_error_instance = 0 
            last_flatten_tree_size = 0 
            last_drafting_batch_size = 0 
            if self.config.check: 
                if (currentlength - last_check) == kernel_size: 
                    self.averagedraftingbatchsize += self.batchsizecount/kernel_size 
                    last_drafting_batch_size = self.batchsizecount/kernel_size 
                    self.batchsizecount = 0 
                    check_flag = True 
                    past_key_values = model_inputs["past_key_values"] 
                    # past_key_values = outputs.past_key_values 
                    # print("cache length {}".format(past_key_values.get_seq_length())) 
                    # print("beforechecking past_key_values shape {}".format(past_key_values.key_cache[0].shape)) 
                    # print("beforechecking extended input_ids shape {}".format(model_inputs["extended_input_ids"].shape)) 
                    # print("length of sequence length {}".format(self.beam[0][0].shape[1])) 
                    past_key_values.mode = "checking" # cache is being roll back 
                    checklength = currentlength - last_check 
                    check_input_ids = model_inputs["extended_input_ids"][:, -checklength:] 
                    # self.flattentreesize += self.merging_tree_into_one_sequence(check_input_ids) 
                    # last_flatten_tree_size = self.merging_tree_into_one_sequence(check_input_ids) 
                    _, self.flattentreesize = self.merging_tree_into_one_sequence2(check_input_ids) 
                    last_flatten_tree_size = self.flattentreesize 
                    # print("input_ids shape {}".format(check_input_ids.shape)) 
                    check_attention_mask = model_inputs["attention_mask"] 
                    # print("check_attention_mask shape {}".format(check_attention_mask.shape)) 
                    check_position_ids = track_position_ids[:, -checklength:] 
                    # print("check_position_ids shape {}".format(check_position_ids.shape)) 
                    check_cache_position = track_cache_position[-checklength :] 
                    # print("check_cache_position shape {}".format(check_cache_position.shape)) 
                    
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
                    full_predicted_likelihood = torch.softmax(full_predicted_logits/0.6, dim = -1) 
                    selected_likelihood = torch.gather(full_predicted_likelihood, dim = -1, index = sparse_predicted_tokens.unsqueeze(-1)).squeeze(-1) 
                    
                    # getting the length of the accepted token 
                    requireallpasses = False 
                    batchdimselection = None 
                    lengthacceptsbig = -1 
                    for i in range(selected_likelihood.shape[0]): 
                        acceptsall = True 
                        lengthaccepts = 0 
                        for k in range(selected_likelihood.shape[1]): 
                            if selected_likelihood[i][k] >= self.config.thr: 
                                lengthaccepts += 1 
                            else: 
                                acceptsall = False 
                                break 
                        if self.verbose: 
                            print(colored("lengthaccepts {}".format(lengthaccepts), "grey"), flush = True) 
                        if lengthaccepts > lengthacceptsbig: 
                            lengthacceptsbig = lengthaccepts 
                            batchdimselection = i 
                        requireallpasses = requireallpasses or acceptsall 
                    
                    outputs_logits_used = outputs.logits[batchdimselection, : (lengthacceptsbig + 1), :].unsqueeze(0) 
                    outputs.logits = outputs_logits_used 
                    step = checklength - (lengthacceptsbig + 1) 
                    
                    input_ids = self.beam[batchdimselection][0] 
                    if self.verbose: 
                        print(colored(self.tokenizer.decode(input_ids[0, last_input_ids_print_pos : last_check + lengthacceptsbig + 1]), "green"), flush = True, end = " ") 
                        if step > 0: 
                            for i in range(step): 
                                print(colored(self.tokenizer.decode(input_ids[0, -step + i]), "red"), flush = True, end = "") 
                                print(colored("({:.2f})".format(selected_likelihood[batchdimselection][lengthacceptsbig + i].item()), "red"), flush = True, end = " ") 
                    
                    # roll back 
                    # past_key_values = self.beam[batchdimselection][2] 
                    past_key_valueslist = GriffinCache.splitcache(model_inputs["past_key_values"]) 
                    past_key_values = past_key_valueslist[batchdimselection] 
                    # print("before rollback past_key_values shape {}".format(past_key_values.key_cache[0].shape)) 
                    # print("before rollback input_ids shape {}".format(input_ids.shape)) 
                    past_key_values.rollback(step) 
                    # print("after rollback past_key_values shape {}".format(past_key_values.key_cache[0].shape)) 
                    past_key_values.mode = "decoding" 
                    last_input_ids_print_pos = last_check + lengthacceptsbig + 1 
                    if step > 0: 
                        input_ids = input_ids[:, : -step] 
                        track_position_ids = track_position_ids[:, :-step] 
                        track_cache_position = track_cache_position[:-step] 
                        # print("after rollback input_ids shape {}".format(input_ids.shape)) 
                    
                    # setup for next iteration 
                    last_check = input_ids.shape[1] 
                    self.num_steps += 1 
                    self.total_steps += lengthacceptsbig + 1 
                    last_total_step = lengthacceptsbig + 1 
                    if lengthacceptsbig != checklength - 1: 
                        self.errorinstance += 1 
                        last_error_instance = 1 
                        self.total_roll_back_length_error += (checklength - 1 - lengthacceptsbig) 
                        # self.roll_back_length_in_error.append(checklength - 1 - lengthacceptsbig) 
                        last_roll_back_length_error = checklength - 1 - lengthacceptsbig 
                    else: 
                        last_error_instance = 0 
                        last_roll_back_length_error = 0 
                    if step == 0: 
                        approve_quit = True 
                    
                    # next_token_logits = outputs.logits[:, -1, :] 
                    next_token_logits = outputs_logits_used[:, -1, :] 
                    next_tokens_scores = logits_processor(input_ids, next_token_logits) 
                    next_tokens = torch.argmax(next_tokens_scores, dim = -1) 
                    input_ids = torch.cat([input_ids, next_tokens[:, None]], dim = -1) 
                    next_token_probs = torch.log_softmax(next_token_logits/0.6, dim = -1) 
                    accumulated_logprob = self.beam[batchdimselection][1] 
                    if self.verbose and self.config.check: 
                        print(colored(self.tokenizer.decode(next_tokens), color = "cyan"), flush = True, end = " ") 
                        print(colored("Choosing {}".format(batchdimselection), "yellow"), flush = True) 
                    # tree update and beam update 
                    self.completed_sequences = [] 
                    # self.beam = [(input_ids, next_token_probs[0][next_tokens.item()] + accumulated_logprob, past_key_values)] 
                    self.beam = [(input_ids, next_token_probs[0][next_tokens.item()], past_key_values)] 
                    if next_tokens.item() == self.tokenizer.eos_token_id: # if large model decodes eos token, we break 
                        break 
                
            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need 

            # pre-process distribution
            # next_tokens_scores = logits_processor(input_ids, next_token_logits) 

            # Store scores, attentions and hidden_states when required 

            # argmax
            # next_tokens = torch.argmax(next_tokens_scores, dim=-1) 
            
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            
            # update generated ids, model inputs, and length for next step
            # input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1) 
            input_ids = self.beam[0][0] 
            next_tokens = input_ids[:, -1] 
            
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            ) 
            '''
            if iteration_count >= 0: 
                print("running in isolation on beam first sequence: {}".format(self.tokenizer.decode(self.beam[0][0][0, initial_len : ]))) 
                cacheintermediate = self.beam[0][2].copy() 
                batch_inputs = [] 
                batch_kvcache = [] 
                batch_attentionmask = [] 
                batch_positionids = [] 
                batch_cacheposition = [] 
                for seq, _, cache in self.beam: 
                    batch_inputs.append(seq[:, -1].unsqueeze(0)) 
                    batch_kvcache.append(cache) 
                    batch_attentionmask.append(torch.ones((1, 1), dtype = input_ids.dtype).to(input_ids.device)) 
                    batch_positionids.append(torch.tensor([[self.beam[0][0].shape[1] - 1]], dtype = input_ids.dtype).to(input_ids.device)) 
                    batch_cacheposition.append(torch.arange(cacheintermediate.seen_tokens, cacheintermediate.seen_tokens + 1, device = input_ids.device, dtype = input_ids.dtype).unsqueeze(0)) 
                # print("cacheintermediate seen tokens {}".format(cacheintermediate.seen_tokens)) 
                # cache_position = torch.arange(cacheintermediate.seen_tokens, cacheintermediate.seen_tokens + 1, device = input_ids.device, dtype = input_ids.dtype) 
                # attention_mask = torch.ones((1, 1), dtype = input_ids.dtype).to(input_ids.device) 
                # position_ids = torch.tensor([[self.beam[0][0].shape[1]]], dtype = input_ids.dtype).to(input_ids.device) 
                # print("input_ids shape {}".format(self.beam[0][0].shape)) 
                # print(self.tokenizer.decode(self.beam[0][0][0]), flush = True) 
                print("type of kv cache {}".format(type(self.beam[0][2]))) 
                # print("input shape {} attention mask {} cache position shape {}".format(torch.cat(batch_inputs, dim = 0).shape, torch.cat(batch_attentionmask, dim = 0).shape, torch.cat(batch_cacheposition, dim = 0).shape)) 
                self.set_inference_mode("partial") 
                outputstwo = self(
                    # input_ids = self.beam[0][0][:, -1].unsqueeze(0).repeat(len(self.beam), 1), 
                    input_ids = torch.cat(batch_inputs, dim = 0), 
                    # input_ids = input_ids[:, : -2], 
                    # attention_mask = torch.ones_like(self.beam[0][0]), 
                    attention_mask = torch.cat(batch_attentionmask, dim = 0), 
                    # position_ids = torch.arange(1, self.beam[0][0].shape[1] + 1, device = input_ids.device, dtype = input_ids.dtype).unsqueeze(0), 
                    position_ids = torch.cat(batch_positionids, dim = 0), 
                    use_cache = True, 
                    past_key_values = GriffinCache.stackcache(batch_kvcache), 
                    return_dict = True, 
                    # cache_position = model_kwargs["cache_position"], 
                    # cache_position = torch.cat(batch_cacheposition, dim = 0), 
                    cache_position = batch_cacheposition[0], 
                    # cache_position = torch.arange(0, self.beam[0][0].shape[1], device = input_ids.device, dtype = input_ids.dtype), 
                ) 
                next_token_logits2 = outputstwo.logits[:, -1, :] 
                for i in range(next_token_logits2.shape[0]): 
                    print("checking whether the input is the same {}".format(self.tokenizer.decode(batch_inputs[i][0])), flush = True) 
                    next_token_logits = next_token_logits2[i].unsqueeze(0) 
                    # next_tokens_scores = logits_processor(input_ids, next_token_logits) 
                    next_tokens_scores = torch.softmax(next_token_logits/0.6, dim = -1) 
                    topk, topkidx = torch.topk(next_tokens_scores, self.k, dim = -1) 
                    topk = topk.squeeze(0) 
                    topkidx = topkidx.squeeze(0) 
                    print("topkidx shape {}, topk shape {}".format(topkidx.shape, topk.shape), flush = True) 
                    for logprob, ids in zip(topk, topkidx): 
                        print(colored("({}, {})".format(self.tokenizer.decode(ids.item()), logprob.item()), color = "yellow"), flush = True) 
                    print() 
                exit(0) 
            ''' 
            if check_flag: 
                model_kwargs["attention_mask"] = model_kwargs["attention_mask"][...,:input_ids.shape[1]] 
            
            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )
                
                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True 

            # stop if we exceed the maximum length
            if stopping_criteria(input_ids, scores): 
                this_peer_finished = True 
                
                if self.config.check: 
                    this_peer_finished = this_peer_finished and check_flag 
                
                if this_peer_finished: 
                    recheckoutcome = False 
                    
                    from transformers.generation.stopping_criteria import MaxLengthCriteria 
                    for stoppingc in stopping_criteria: 
                        if isinstance(stoppingc, MaxLengthCriteria): 
                            recheckoutcome = recheckoutcome or stoppingc(input_ids, scores) 
                        
                        if isinstance(stoppingc, MultiTokenEOSCriteria): 
                            stoppingc.done_tracker = [False] * input_ids.shape[0] 
                            recheckoutcome = recheckoutcome or stoppingc(input_ids, scores) 
                    if self.config.check: 
                        this_peer_finished = recheckoutcome 

            else: 
                this_peer_finished = False 

            # print("stopping_criteria {}".format(stopping_criteria)) 
            
            if this_peer_finished and not synced_gpus: 
                want_to_quit = True 
                if approve_quit or (not self.config.check): 
                    break 
        
        if self.config.griffin: 
            self.reset_states() 
        track_position_ids = None 
        track_cache_position = None 
        
        if streamer is not None:
            streamer.end()
        
        self.totalgenerationlength += input_ids.shape[1] - initial_len 
        self.batchsizecount = 0 
        if self.config.check: 
            self.rollbacklastchunkstatistic(last_total_step, last_error_instance, last_roll_back_length_error, last_flatten_tree_size, last_average_drafting_batch_size = last_drafting_batch_size) 
        # print("total generation length {}".format(self.totalgenerationlength)) 
        # print(self.tokenizer.decode(input_ids[0][initial_len : ])) 

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
                    scores=None, 
                    logits=None, 
                    attentions=None, 
                    hidden_states=None, 
                    past_key_values=model_kwargs.get("past_key_values"),
                ) 
        else: 
            return input_ids
