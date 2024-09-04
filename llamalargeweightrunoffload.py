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
from cache import StaticCachePipa 
import torch.nn.functional as F 
import numpy as np 

from transformers.generation.logits_process import (
    LogitsProcessorList, 
) 

from transformers.generation.stopping_criteria import (
    StoppingCriteriaList, 
    validate_stopping_criteria, 
) 

from transformers import LlamaForCausalLM 

from transformers.generation.utils import GenerateOutput, GenerateDecoderOnlyOutput, GenerateNonBeamOutput, GenerateEncoderDecoderOutput

from time import time 
from termcolor import colored 
import torch._dynamo 

import gc 

# from transformers.utils import (
#     add_start_docstrings,
#     add_start_docstrings_to_model_forward,
#     is_flash_attn_2_available,
#     is_flash_attn_greater_or_equal_2_10,
#     logging,
#     replace_return_docstrings,
# )

# if is_flash_attn_2_available():
#     from flash_attn import flash_attn_func, flash_attn_varlen_func
#     from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa 

class LayerBuffer: 
    def __init__(self, config, sparsitylevel = 0.5): 
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        
        # attention weights 
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias).to(torch.bfloat16).to("cuda:0") 
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias).to(torch.bfloat16).to("cuda:0") 
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias).to(torch.bfloat16).to("cuda:0") 
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias).to(torch.bfloat16).to("cuda:0") 
        
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size 
        
        # mlp layer 
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False).to(torch.bfloat16).to("cuda:0") 
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False).to(torch.bfloat16).to("cuda:0") 
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False).to(torch.bfloat16).to("cuda:0") 
        
        # GRIFFIN layer 
        self.gate_proj_reduced = nn.Linear(self.hidden_size, int(self.intermediate_size * sparsitylevel), bias = False).to(torch.bfloat16).to("cuda:0") 
        self.up_proj_reduced = nn.Linear(self.hidden_size, int(self.intermediate_size * sparsitylevel), bias = False).to(torch.bfloat16).to("cuda:0") 
        self.down_proj_reduced = nn.Linear(int(self.intermediate_size * sparsitylevel), self.hidden_size, bias = False).to(torch.bfloat16).to("cuda:0") 
    
    def copy(self, layer, griffin = False): 
        self.q_proj.weight.data.copy_(layer.self_attn.q_proj.weight.data, non_blocking = True) 
        self.k_proj.weight.data.copy_(layer.self_attn.k_proj.weight.data, non_blocking = True) 
        self.v_proj.weight.data.copy_(layer.self_attn.v_proj.weight.data, non_blocking = True) 
        self.o_proj.weight.data.copy_(layer.self_attn.o_proj.weight.data, non_blocking = True) 
        
        if not griffin: 
            self.gate_proj.weight.data.copy_(layer.mlp.gate_proj.weight.data, non_blocking = True) 
            self.up_proj.weight.data.copy_(layer.mlp.up_proj.weight.data, non_blocking = True) 
            self.down_proj.weight.data.copy_(layer.mlp.down_proj.weight.data, non_blocking = True) 
        else: 
            self.gate_proj_reduced.weight.data.copy_(layer.mlp.gate_proj_reduced.weight.data, non_blocking = True) 
            self.up_proj_reduced.weight.data.copy_(layer.mlp.up_proj_reduced.weight.data, non_blocking = True) 
            self.down_proj_reduced.weight.data.copy_(layer.mlp.down_proj_reduced.weight.data, non_blocking = True) 

class LlamaLargeOffload: 
    def __init__(self, config, model): 
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False).to(torch.bfloat16).to("cuda:0") 
        self.lm_head.weight.data.copy_(model.lm_head.weight.data, non_blocking = False) 
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id).to(torch.bfloat16).to("cuda:0") # using embedding 
        self.embed_tokens.weight.data.copy_(model.model.embed_tokens.weight.data, non_blocking = False) 
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(torch.bfloat16).to("cuda:0") 
        self.norm.weight.data.copy_(model.model.norm.weight.data, non_blocking = False) 
        self.stream = torch.cuda.Stream(device = "cuda:0") 
        causal_mask = torch.full((config.max_position_embeddings, config.max_position_embeddings), fill_value=1).to("cuda:0") 
        # self.register_buffer("causal_mask", torch.triu(causal_mask, diagonal=1), persistent=False)
        self.causal_mask = torch.triu(causal_mask, diagonal=1)
        self.act_fn = F.silu 
        self.griffin = False 
        
        self.all_layers_inputlayernorm = [] 
        self.all_layers_postattentionlayernorm = [] 
        self.all_layers_rotaryemb = [] 
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta 
        self.num_hidden_layers = config.num_hidden_layers 
        
        for layer_idx in range(config.num_hidden_layers): 
            input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(torch.bfloat16).to("cuda:0") 
            input_layernorm.weight.data.copy_(model.model.layers[layer_idx].input_layernorm.weight.data, non_blocking = False) 
            post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps = config.rms_norm_eps).to(torch.bfloat16).to("cuda:0") 
            post_attention_layernorm.weight.data.copy_(model.model.layers[layer_idx].post_attention_layernorm.weight.data, non_blocking = False) 
            rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings = self.max_position_embeddings, base = self.rope_theta).to(torch.bfloat16).to("cuda:0") 
            self.all_layers_inputlayernorm.append(input_layernorm) 
            self.all_layers_postattentionlayernorm.append(post_attention_layernorm) 
            self.all_layers_rotaryemb.append(rotary_emb) 
        
        max_seq_len = 1024 
        # self.past_key_values = StaticCachePipa(config = self.config, max_batch_size = 1, max_cache_len = max_seq_len, device = "cuda:0", dtype = torch.bfloat16) 
        self.past_key_values = None 
        
        self.layerbuffer = [LayerBuffer(config) for _ in range(2)] 
    
    def set_inference_mode(self, mode): 
        if mode == "full": 
            self.griffin = False 
        elif mode == "partial": 
            self.griffin = True 
        else: 
            raise ValueError("Invalid mode") 
        
    def _update_causal_mask(self, attention_mask, input_tensor):

        batch_size, seq_length = input_tensor.shape[:2]
        dtype = input_tensor.dtype
        device = input_tensor.device

        # support going beyond cached `max_position_embedding`
        if seq_length > self.causal_mask.shape[-1]:
            causal_mask = torch.full((2 * self.causal_mask.shape[-1], 2 * self.causal_mask.shape[-1]), fill_value=1)
            self.register_buffer("causal_mask", torch.triu(causal_mask, diagonal=1), persistent=False)

        if hasattr(self, "causal_mask"):  # we use the current dtype to avoid any overflows
            causal_mask = (
                self.causal_mask[None, None, :, :].repeat(batch_size, 1, 1, 1).to(dtype) * torch.finfo(dtype).min
            )
        else:
            mask = torch.full(
                (self.config.max_position_embeddings, self.config.max_position_embeddings),
                fill_value=torch.finfo(dtype).min,
            )
            causal_mask = torch.triu(mask, diagonal=1)

        causal_mask = causal_mask.to(dtype=dtype, device=device)
        if attention_mask is not None and attention_mask.dim() == 2:
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
            causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(
                padding_mask, torch.finfo(dtype).min
            ) 

        return causal_mask
    
    def forward(
        self, 
        input_ids, 
        attention_mask, 
        position_ids, 
        use_cache, 
        past_key_values, 
        cache_position, 
        model, 
        **kwargs): 
        if past_key_values is None: 
            raise ValueError("past_key_values is None") 
        
        inputs_embeds = self.embed_tokens(input_ids) 
        
        # embeddingt = model.model.embed_tokens.to("cuda:0") 
        # inputs_embeds = embeddingt(input_ids) 
        if position_ids is None:
            # for verification
            position_ids = cache_position.unsqueeze(0) 
        hidden_states = inputs_embeds 
        # print("hidden_states dtype {}".format(hidden_states.dtype)) 
        
        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds) # for flash_attention_2, this line should be one or two if statements 
        
        self.layerbuffer[0].copy(model.model.layers[0], self.griffin) 
        
        for layer_idx in range(model.config.num_hidden_layers): 
            torch.cuda.synchronize() 
            with torch.cuda.stream(self.stream): 
                hidden_states = self.run_layer(
                    hidden_states = hidden_states, 
                    attention_mask = causal_mask, 
                    position_ids = position_ids, 
                    past_key_values = past_key_values, 
                    use_cache = use_cache, 
                    cache_position = cache_position, 
                    layeridx = layer_idx, 
                    idx_selection = layer_idx % 2, 
                    griffin = self.griffin, 
                    **kwargs, 
                ) 
            if layer_idx != self.num_hidden_layers - 1: 
                self.layerbuffer[(layer_idx + 1) % 2].copy(model.model.layers[layer_idx + 1], self.griffin) 
            torch.cuda.synchronize() 
        
        # for layer_idx in range(model.config.num_hidden_layers): 
        #     layer = model.model.layers[layer_idx].to("cuda:0") 
        #     hidden_states = layer(
        #         hidden_states = hidden_states, 
        #         attention_mask = causal_mask, 
        #         position_ids = position_ids, 
        #         use_cache = use_cache, 
        #         cache_position = cache_position, 
        #         past_key_value = past_key_values, 
        #     ) 
            
        #     hidden_states = hidden_states[0] 
        
        hidden_states = self.norm(hidden_states) 
        logits = self.lm_head(hidden_states) 

        # norm = model.model.norm.to("cuda:0") 
        # lm_head = model.lm_head.to("cuda:0") 
        # hidden_states = norm(hidden_states) 
        # logits = lm_head(hidden_states) 
        logits = logits.float() 
        
        return logits 

    def run_layer(
        self, 
        hidden_states, 
        attention_mask, 
        position_ids, 
        past_key_values, 
        use_cache, 
        cache_position, 
        layeridx, 
        idx_selection, 
        griffin, 
        **kwargs, 
    ): 
        residual = hidden_states 
        hidden_states = self.all_layers_inputlayernorm[layeridx](hidden_states) 
        
        bsz, q_len, _ = hidden_states.size()
        query_states = self.layerbuffer[idx_selection].q_proj(hidden_states) 
        key_states = self.layerbuffer[idx_selection].k_proj(hidden_states) 
        value_states = self.layerbuffer[idx_selection].v_proj(hidden_states) 
        
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = self.all_layers_rotaryemb[layeridx](value_states, position_ids) 
        
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin) 
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, layeridx, cache_kwargs) 
        key_states = repeat_kv(key_states.clone(), self.num_key_value_groups) 
        value_states = repeat_kv(value_states.clone(), self.num_key_value_groups) 
        
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))/math.sqrt(self.head_dim) 
        if attention_mask is not None:  # no matter the length, we just slice it
            # causal_mask = attention_mask[:, :, :, : key_states.shape[-2]] 
            causal_mask = attention_mask[:, :, cache_position, : key_states.shape[-2]] 
            attn_weights = attn_weights + causal_mask
        
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        # del key_states 
        # del value_states 
        # gc.collect() 
        # torch.cuda.empty_cache() 
        # torch.cuda.synchronize() 

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        
        attn_output = self.layerbuffer[idx_selection].o_proj(attn_output) 
        hidden_states = residual + attn_output 
        
        residual = hidden_states 
        hidden_states = self.all_layers_postattentionlayernorm[layeridx](hidden_states) 
        
        if not griffin: 
            int_states = self.act_fn(self.layerbuffer[idx_selection].gate_proj(hidden_states)) * self.layerbuffer[idx_selection].up_proj(hidden_states) 
            down_proj = self.layerbuffer[idx_selection].down_proj(int_states) 
        else: 
            int_states = self.act_fn(self.layerbuffer[idx_selection].gate_proj_reduced(hidden_states)) * self.layerbuffer[idx_selection].up_proj_reduced(hidden_states) 
            down_proj = self.layerbuffer[idx_selection].down_proj_reduced(int_states) 
        
        hidden_states = residual + down_proj 
        
        return hidden_states 

def run_inference(
    input_ids,
    model, 
    device_model, 
    eos_token_id, 
    kernel_size = 16, 
    tokenizer = None, 
    **generation_kwargs
): 
    # model is on CPU; device_model is on GPU; 
    initial_len = input_ids.shape[1] 
    
    max_seq_len = 1024 
    
    with torch.no_grad(): 
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
        
        past_key_values = StaticCachePipa(config = model.config, max_batch_size = 1, max_cache_len = max_seq_len, device = "cpu", dtype = torch.bfloat16) 
        attention_mask = generation_kwargs["attention_mask"] 
        position_ids = attention_mask[:, : input_ids.shape[1]].long().cumsum(-1) - 1 
        position_ids.masked_fill_(attention_mask[:, : input_ids.shape[1]] == 0, 1) 
        cache_position = torch.arange(0, position_ids.shape[1], device = input_ids.device, dtype = position_ids.dtype) 
        attention_mask_static = torch.zeros((generation_kwargs["attention_mask"].shape[0], max_seq_len)).to(input_ids.device).to(generation_kwargs["attention_mask"].dtype) 
        # attention_mask_static = torch.zeros((model_kwargs["attention_mask"].shape[0], 1024)).to(input_ids.device).to(model_kwargs["attention_mask"].dtype) 
        attention_mask_static[:, : generation_kwargs["attention_mask"].shape[1]] = generation_kwargs["attention_mask"] 
        attention_mask = attention_mask_static # attentionmask is shared 
        
        # parameters for sparse 
        executinginputidssparse = torch.empty((1, 1)).to(input_ids.device, torch.long) 
        executingcachepositionsparse = torch.empty((1,)).to(input_ids.device, torch.long) 
        outputs = torch.empty((1, 1, model.config.vocab_size), device = input_ids.device, dtype = torch.bfloat16).cpu() 
        
        # parameters for full 
        executinginputidsfull = torch.empty((1, kernel_size)).to(input_ids.device, torch.long) 
        executingcachepositionfull = torch.empty((kernel_size,)).to(input_ids.device, torch.long) 
        outputsfull = torch.empty((1, model.config.kernel_size, model.config.vocab_size), device = input_ids.device, dtype = torch.bfloat16) 
        
        assert input_ids.shape[0] == 1 
        
        model.set_inference_mode("full") 
        
        outputs = model(
            input_ids = input_ids.cpu(), 
            attention_mask = attention_mask.cpu(), 
            position_ids = None, 
            use_cache = True, 
            past_key_values = past_key_values, 
            cache_position = cache_position.cpu(), 
        ) 
        
        # outputs = model(
        #     input_ids = input_ids, 
        #     attention_mask = attention_mask, 
        #     position_ids = None, 
        #     use_cache = True, 
        #     past_key_values = past_key_values, 
        #     cache_position = cache_position, 
        # ) 
        # outputs = outputs.to("cuda:0") 
        next_token_logits = outputs.logits[:, -1, :] 
        next_tokens = torch.argmax(next_token_logits, dim=-1) 
        # update generated ids, model inputs, and length for next step 
        
        input_ids = input_ids.cpu() 
        
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1) 
        print(tokenizer.decode(input_ids[0])) 
        length_inputids = input_ids.shape[1] 
        staticinputids = torch.zeros((input_ids.shape[0], input_ids.shape[1] + 256)).to("cuda:0").to(input_ids.dtype) 
        staticinputids[:, : input_ids.shape[1]] = input_ids 
        attention_mask[0, length_inputids - 1] = 1 
        
        cache_position = torch.arange(past_key_values.get_seq_length(), past_key_values.get_seq_length() + 1, device = cache_position.device, dtype = cache_position.dtype) 
        model.memorysetonhost() 
        
        # preparing graph 
        executinginputidssparse.copy_(input_ids[:, -1].unsqueeze(0)) 
            
        executingcachepositionsparse.copy_(cache_position) 

        device_model.set_inference_mode("partial") 
        
        past_key_values.move_to_device("cuda:0") 
        cache_positionadder = torch.arange(0, kernel_size, device = cache_position.device, dtype = cache_position.dtype) 
        
        torch.cuda.synchronize() 
        starttime = time() 
        '''
        for i in range(20): 
                
            device_model.set_inference_mode("partial") 
            outputs = device_model.forward(
                input_ids = executinginputidssparse, 
                attention_mask = attention_mask, 
                position_ids = None, 
                use_cache = True, 
                past_key_values = past_key_values, 
                cache_position = executingcachepositionsparse, 
                model = model, 
            ) 
            
            # outputs = model(
            #     input_ids = executinginputidssparse, 
            #     attention_mask = attention_mask, 
            #     position_ids = None, 
            #     use_cache = True, 
            #     past_key_values = past_key_values, 
            #     cache_position = executingcachepositionsparse, 
            # ) 
            
            next_token_logits = outputs[:, -1, :] 
            # next_token_logits = outputs.logits[:, -1, :] 
            next_token = torch.argmax(next_token_logits, dim = -1) 
            print(tokenizer.decode(next_token[0])) 
            
            staticinputids[:, length_inputids] = next_token 
            length_inputids += 1 
            attention_mask[0, length_inputids - 1] = 1 
            executinginputidssparse.copy_(next_token.unsqueeze(0)) 
            executingcachepositionsparse.copy_(executingcachepositionsparse + 1) 
        '''
        
        while (length_inputids - initial_len) < 80: 
            if (length_inputids - last_check) != kernel_size: 
                
                device_model.set_inference_mode("partial") 
                outputs = device_model.forward(
                    input_ids = executinginputidssparse, 
                    attention_mask = attention_mask, 
                    position_ids = None, 
                    use_cache = True, 
                    past_key_values = past_key_values, 
                    cache_position = executingcachepositionsparse, 
                    model = model, 
                ) 
                
                # outputs = model(
                #     input_ids = executinginputidssparse, 
                #     attention_mask = attention_mask, 
                #     position_ids = None, 
                #     use_cache = True, 
                #     past_key_values = past_key_values, 
                #     cache_position = executingcachepositionsparse, 
                # ) 
            
                next_token_logits = outputs[:, -1, :] 
                # next_token_logits = outputs.logits[:, -1, :] 
                next_token = torch.argmax(next_token_logits, dim = -1) 
                print(tokenizer.decode(next_token), flush = True, end = "") 
                
                staticinputids[:, length_inputids] = next_token 
                length_inputids += 1 
                attention_mask[0, length_inputids - 1] = 1 
                executinginputidssparse.copy_(next_token.unsqueeze(0)) 
                executingcachepositionsparse.copy_(executingcachepositionsparse + 1) 
            else: 
                executinginputidsfull.copy_(staticinputids[:, length_inputids - kernel_size : length_inputids]) 
                executingcachepositionfull.copy_(cache_positionadder + length_inputids - kernel_size) 
                device_model.set_inference_mode("full") 
                outputsfull = device_model.forward(
                    input_ids = executinginputidsfull, 
                    attention_mask = attention_mask, 
                    position_ids = None, 
                    use_cache = True, 
                    past_key_values = past_key_values, 
                    cache_position = executingcachepositionfull, 
                    model = model, 
                ) 
                sparse_predicted_tokens = staticinputids[:, length_inputids - kernel_size + 1 : length_inputids] 
                full_predicted_tokens = torch.softmax(outputsfull[:, : -1, :]/0.6, dim = -1) 
                selected_likelihood = torch.gather(full_predicted_tokens, dim = -1, index = sparse_predicted_tokens.unsqueeze(-1)).squeeze(-1).squeeze(0) 
                
                batchdimselection = None 
                thresholdgreatr = selected_likelihood >= model.config.thr 
                lengthaccepts = torch.cumprod(thresholdgreatr.to(torch.int64), dim = 0).sum().item() 
                
                outputsfullback = outputsfull[:, lengthaccepts, :] 
                model.num_steps += 1 
                model.total_steps += lengthaccepts + 1 
                
                step = kernel_size - (lengthaccepts + 1) 
                executingcachepositionsparse.copy_(executingcachepositionsparse - step + 1) 
                attention_mask[0, length_inputids - step : length_inputids] = 0 
                length_inputids -= step 
                
                next_token = torch.argmax(outputsfullback, dim = -1) 
                staticinputids[:, length_inputids] = next_token 
                last_check = length_inputids 
                
                length_inputids += 1 
                attention_mask[0, length_inputids - 1] = 1 
                executinginputidssparse.copy_(next_token.unsqueeze(0)) 
                input_ids = staticinputids[:, : length_inputids] 
            
        
        torch.cuda.synchronize() 
        endtime = time() 
        # print("time taken is {}".format((endtime - starttime)/20)) 
        # exit(0) 
        
        print("acceptance length on average {}".format(model.total_steps/model.num_steps)) 
        print("time taken for a single forward pass {}".format((endtime - starttime))) 
        model.generationtime += (endtime - starttime) 
        
        model.totalgenerationlength += (length_inputids - initial_len - 1) 
        torch.cuda.synchronize() 
        model.reset_states() 
        print("after the generation allocated memory {}".format(torch.cuda.memory_allocated(0))) 
        print("per token generation time {}".format(model.generationtime/model.totalgenerationlength)) 
    
        return staticinputids[:, : length_inputids] 
