import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from transformers import LlamaForCausalLM 
import math
import warnings
from typing import List, Optional, Tuple, Union
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union 

from transformers.generation.utils import GenerateOutput, GenerateDecoderOnlyOutput, GenerateNonBeamOutput, GenerateEncoderDecoderOutput 

from transformers.generation.logits_process import (
    LogitsProcessorList, 
) 
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList, 
    validate_stopping_criteria, 
) 
from cache import StaticCacheSDPA 

from lm_eval.models.utils import MultiTokenEOSCriteria 

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

class LlamaForCausalLMPipa(LlamaForCausalLM): 
    def __init__(self, config): 
        super().__init__(config) 
        self.total_steps = 0 
        self.num_steps = 0 
        self.num_sentences = 0 
        self.totalgenerationlength = 0 
        self.total_roll_back_length_error = 0 
        self.total_roll_back_length_error = 0 
        self.roll_back_length_in_error = [] 
        self.errorinstance = 0 
        self.verbose = False 
        
        from transformers import AutoTokenizer 
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct") 
        if self.verbose: 
            assert self.tokenizer is not None 
    
    def updatestatistic(self): 
        self.num_steps = 0 
        self.num_sentence = 0 
        self.total_steps = 0 
        self.totalgenerationlength = 0 
        self.total_roll_back_length_error = 0 
        self.roll_back_length_in_error = [] 
        self.errorinstance = 0 
    
    def set_inference_mode(self, mode): 
        for layer in self.model.layers: 
            layer.mlp.inference_mode = mode 
    
    def reset_states(self): 
        for layer in self.model.layers: 
            layer.mlp.reset_stats() 
    
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
    ): 
        
        initial_len = input_ids.shape[1] 
        
        max_seq_len = 2048 
        
        print("input_ids shape {}".format(input_ids.shape)) 
        kernel_size = self.config.kernel_size 
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList() 
        for stoppingcrition in stopping_criteria: 
            if isinstance(stoppingcrition, MultiTokenEOSCriteria): 
                stoppingcrition.sequence_id_len = len(stoppingcrition.sequence_ids) + self.config.kernel_size 
        
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
        
        past_key_values = StaticCacheSDPA(config = self.config, max_batch_size = 1, max_cache_len = max_seq_len, device = input_ids.device, dtype = torch.bfloat16) 
        attention_mask = model_kwargs["attention_mask"] 
        position_ids = attention_mask[:, : input_ids.shape[1]].long().cumsum(-1) - 1 
        position_ids.masked_fill_(attention_mask[:, : input_ids.shape[1]] == 0, 1) 
        cache_position = torch.arange(0, position_ids.shape[1], device = input_ids.device, dtype = position_ids.dtype) 
        attention_mask_static = torch.zeros((model_kwargs["attention_mask"].shape[0], max_seq_len)).to(input_ids.device).to(model_kwargs["attention_mask"].dtype) 
        # attention_mask_static = torch.zeros((model_kwargs["attention_mask"].shape[0], 1024)).to(input_ids.device).to(model_kwargs["attention_mask"].dtype) 
        attention_mask_static[:, : model_kwargs["attention_mask"].shape[1]] = model_kwargs["attention_mask"] 
        attention_mask = attention_mask_static # attentionmask is shared 
        
        # parameters for sparse 
        executinginputidssparse = torch.empty((1, 1)).to(input_ids.device, torch.long) 
        executingcachepositionsparse = torch.empty((1,)).to(input_ids.device, torch.long) 
        outputs = torch.empty((1, 1, self.config.vocab_size), device = input_ids.device, dtype = torch.bfloat16) 
        
        # parameters for full 
        executinginputidsfull = torch.empty((1, self.config.kernel_size)).to(input_ids.device, torch.long) 
        executingcachepositionfull = torch.empty((self.config.kernel_size,)).to(input_ids.device, torch.long) 
        outputsfull = torch.empty((1, self.config.kernel_size, self.config.vocab_size), device = input_ids.device, dtype = torch.bfloat16) 
        
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
        next_token_logits = outputs.logits[:, -1, :] 
        next_tokens = torch.argmax(next_token_logits, dim=-1) 
        # update generated ids, model inputs, and length for next step 
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1) 
        length_inputids = input_ids.shape[1] 
        staticinputids = torch.zeros((input_ids.shape[0], input_ids.shape[1] + 256)).to(input_ids.device).to(input_ids.dtype) 
        staticinputids[:, : input_ids.shape[1]] = input_ids 
        attention_mask[0, length_inputids - 1] = 1 
        
        cache_position = torch.arange(past_key_values.get_seq_length(), past_key_values.get_seq_length() + 1, device = cache_position.device, dtype = cache_position.dtype) 
        
        # preparing graph 
        executinginputidssparse.copy_(input_ids[:, -1].unsqueeze(0)) 
            
        executingcachepositionsparse.copy_(cache_position) 
        self.set_inference_mode("partial") 
        cache_positionadder = torch.arange(0, self.config.kernel_size, device = cache_position.device, dtype = cache_position.dtype) 
        
        while True: 
            if (length_inputids - last_check) != kernel_size: 
                # second, we run the model 
                outputs = self( 
                    input_ids = executinginputidssparse, 
                    attention_mask = attention_mask, 
                    position_ids = None, 
                    use_cache = True, 
                    past_key_values = past_key_values, 
                    cache_position = executingcachepositionsparse, 
                    # inputs_embeds = input_embeds, 
                ) 
                
                next_token_logits = outputs.logits[:, -1, :] 
                next_token = torch.argmax(next_token_logits, dim = -1) 
                
                staticinputids[:, length_inputids] = next_token 
                length_inputids += 1 
                attention_mask[0, length_inputids - 1] = 1 
                executinginputidssparse.copy_(next_token.unsqueeze(0)) 
                executingcachepositionsparse.copy_(executingcachepositionsparse + 1) 
            else: 
                executinginputidsfull.copy_(staticinputids[:, length_inputids - kernel_size : length_inputids]) 
                executingcachepositionfull.copy_(cache_positionadder + length_inputids - kernel_size) 
                outputsfull = self(
                    input_ids = executinginputidsfull, 
                    attention_mask = attention_mask, 
                    position_ids = None, 
                    use_cache = True, 
                    past_key_values = past_key_values, 
                    cache_position = executingcachepositionfull, 
                ) 
                sparse_predicted_tokens = staticinputids[:, length_inputids - kernel_size + 1 : length_inputids] 
                full_predicted_likelihood = torch.softmax(outputsfull[:, : -1, :]/0.6, dim = -1) 
                selected_likelihood = torch.gather(full_predicted_likelihood, dim = -1, index = sparse_predicted_tokens.unsqueeze(-1)).squeeze(-1).squeeze(0) # I didn't change this part, think about how to make it better 
                
                batchdimselection = None 
                # another way to write this 
                thresholdgreatr = selected_likelihood >= self.config.thr 
                lengthaccepts = torch.cumprod(thresholdgreatr.to(torch.int64), dim = 0).sum().item() 
                
                outputsfullback = outputsfull[:, lengthaccepts, :] 
                self.num_steps += 1 
                self.total_steps += lengthaccepts + 1 
                
                step = kernel_size - (lengthaccepts + 1) 
                executingcachepositionsparse.copy_(executingcachepositionsparse - step + 1) 
                attention_mask[0, length_inputids - step : length_inputids] = 0 
                length_inputids -= step 
                
                last_check = length_inputids 
                next_token = torch.argmax(outputsfullback, dim = -1) 
                staticinputids[:, length_inputids] = next_token 
                length_inputids += 1 
                attention_mask[0, length_inputids - 1] = 1 
                executinginputidssparse.copy_(next_token.unsqueeze(0)) 
                input_ids = staticinputids[:, : length_inputids] 
                if step == 0: 
                    approve_quit = True 
            
                if stopping_criteria(input_ids, scores = None): 
                    this_peer_finished = True 
                    
                    recheckoutcome = False 
                    from transformers.generation.stopping_criteria import MaxLengthCriteria 
                    for stoppingc in stopping_criteria: 
                        if isinstance(stoppingc, MaxLengthCriteria): 
                            recheckoutcome = recheckoutcome or stoppingc(input_ids, scores = None) 
                        
                        if isinstance(stoppingc, MultiTokenEOSCriteria): 
                            stoppingc.done_tracker = [False] * input_ids.shape[0] 
                            recheckoutcome = recheckoutcome or stoppingc(input_ids, scores = None) 
                    this_peer_finished = recheckoutcome 
                else: 
                    this_peer_finished = False 
                
                if this_peer_finished and not synced_gpus: 
                    want_to_quit = True 
                    if approve_quit or (not self.config.check): 
                        break 
        
        print("acceptance length on average {}".format(self.total_steps/self.num_steps)) 
        
        if self.config.griffin: 
            self.reset_states() 
        input_ids = staticinputids[:, : length_inputids] 
        
        self.totalgenerationlength += (length_inputids - initial_len - 1) 
        
        return input_ids 
            
