from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from transformers.cache_utils import Cache
import torch

class GriffinCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        # self.seen_token_internal = 0 
        self.mode = "decoding"

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache) 
    
    def adding_one_entry(
        self, 
    ) -> Tuple[torch.Tensor, torch.Tensor]: 
        assert len(self) > 0 
        self.seen_tokens += 1 
        for i in range(len(self.key_cache)):
            self.key_cache[i] = torch.cat([self.key_cache[i], torch.zeros_like(self.key_cache[i][..., 0, :]).unsqueeze(dim = -2)], dim = -2) 
            
        for i in range(len(self.value_cache)):
            self.value_cache[i] = torch.cat([self.value_cache[i], torch.zeros_like(self.value_cache[i][..., 0, :]).unsqueeze(dim = -2)], dim = -2) 

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        
        if self.mode == "decoding":
            if layer_idx == 0:
                self.seen_tokens += key_states.shape[-2] 
                # self.seen_token_internal += key_states.shape[-2] 

            # Update the cache
            if len(self.key_cache) <= layer_idx:
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

            return self.key_cache[layer_idx], self.value_cache[layer_idx]
        elif self.mode == "checking":
            kernel_size = key_states.shape[-2]
            self.key_cache[layer_idx][...,-kernel_size:,:] = key_states
            self.value_cache[layer_idx][...,-kernel_size:,:] = value_states
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        else:
            kernel_size = key_states.shape[-2]
            key = self.key_cache[layer_idx].clone()
            value = self.value_cache[layer_idx].clone()
            key[...,-kernel_size:,:] = key_states
            value[...,-kernel_size:,:] = value_states
            return key, value

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "GriffinCache":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache
    
    @staticmethod 
    def stackcache(previouscache: List["GriffinCache"]) -> "GriffinCache": 
        # check seen_tokens 
        previous_seen_token = None 
        previous_mode = None 
        for cache in previouscache: 
            assert isinstance(cache, GriffinCache) 
            if previous_seen_token is None: 
                previous_seen_token = cache.seen_tokens 
                previous_mode = cache.mode 
            else: 
                assert cache.seen_tokens == previous_seen_token 
                assert cache.mode == previous_mode 
        
        new_key_cache = [] 
        new_value_cache = [] 
        for layer_idx in range(len(previouscache[0].key_cache)): 
            k_layer = [] 
            v_layer = [] 
            for cache in previouscache: 
                k_layer.append(cache.key_cache[layer_idx]) 
                v_layer.append(cache.value_cache[layer_idx]) 
            new_key_cache.append(torch.cat(k_layer, dim = 0)) 
            new_value_cache.append(torch.cat(v_layer, dim = 0)) 
        
        new_cache = GriffinCache() # new cache 
        new_cache.key_cache = new_key_cache 
        new_cache.value_cache = new_value_cache 
        new_cache.seen_tokens = previous_seen_token 
        new_cache.mode = previous_mode 
        
        return new_cache 
    
    @staticmethod 
    def splitcache(cache: "GriffinCache") -> List["GriffinCache"]: 
        new_cache = [] 
        for i in range(len(cache.key_cache[0])): 
            cacheitemp = GriffinCache() 
            cacheitemp.key_cache = [k[i].unsqueeze(dim = 0) for k in cache.key_cache] 
            cacheitemp.value_cache = [v[i].unsqueeze(dim = 0) for v in cache.value_cache] 
            cacheitemp.seen_tokens = cache.seen_tokens 
            cacheitemp.mode = cache.mode 
            new_cache.append(cacheitemp) 
        return new_cache 
    
    def copy(self): 
        new_cache = GriffinCache() 
        new_cache.key_cache = [k.clone() for k in self.key_cache] 
        new_cache.value_cache = [v.clone() for v in self.value_cache] 
        new_cache.seen_tokens = self.seen_tokens 
        new_cache.mode = self.mode 
        return new_cache 
                    
    def rollback(self, steps:int):
        
        if steps > 0:
            self.seen_tokens  = self.seen_tokens - steps 
            # self.seen_token_internal = self.seen_token_internal - steps 
            for i in range(len(self.key_cache)):
                self.key_cache[i] = self.key_cache[i][...,:-steps,:]
                
            for i in range(len(self.value_cache)):
                self.value_cache[i] = self.value_cache[i][...,:-steps,:]
    
