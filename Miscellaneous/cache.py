from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from transformers.cache_utils import Cache
import torch 
import time 

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

class StaticCache(Cache):
    """
    Static Cache class to be used with `torch.compile(model)`.

    Parameters:
        config (`PretrainedConfig):
            The configuration file defining the shape-related attributes required to initialize the static cache.
        max_batch_size (`int`):
            The maximum batch size with which the model will be used.
        max_cache_len (`int`):
            The maximum sequence length with which the model will be used.
        device (`torch.device`):
            The device on which the cache should be initialized. Should be the same as the layer.
        dtype (*optional*, defaults to `torch.float32`):
            The default `dtype` to use when initializing the layer.
    """

    def __init__(self, config, max_batch_size: int, max_cache_len: int, device, dtype=None): 
        super().__init__()
        self.max_batch_size = max_batch_size
        self.max_cache_len = config.max_position_embeddings if max_cache_len is None else max_cache_len
        # Some model define a custom `head_dim` != config.hidden_size // config.num_attention_heads
        self.head_dim = (
            config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
        ) 
        self.num_layers = config.num_hidden_layers 

        self.dtype = dtype if dtype is not None else torch.float32
        self.num_key_value_heads = (
            config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        )

        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        # cache_shape = (max_batch_size, self.num_key_value_heads, self.max_cache_len, self.head_dim) 
        cache_shape = (max_batch_size, self.max_cache_len, self.num_key_value_heads, self.head_dim) 
        for _ in range(config.num_hidden_layers):
            # Note: `mark_static_address` is used to tag the cache as an fixed data pointer, preventing cuda graph
            # breaks when updating the cache.
            new_layer_key_cache = torch.zeros(cache_shape, dtype=self.dtype, device=device)
            new_layer_value_cache = torch.zeros(cache_shape, dtype=self.dtype, device=device)
            torch._dynamo.mark_static_address(new_layer_key_cache)
            torch._dynamo.mark_static_address(new_layer_value_cache)
            self.key_cache.append(new_layer_key_cache)
            self.value_cache.append(new_layer_value_cache) 
        
        self.cachelength = 0 

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        It is VERY important to index using a tensor, otherwise you introduce a copy to the device.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The `StaticCache` needs the `cache_position` input
                to know how where to write in the cache.

        Return:
            A tuple containing the updated key and value states.
        """
        # cache_position = cache_kwargs.get("cache_position") 
        # self.key_cache[layer_idx][:, :, self.cachelength:self.cachelength + key_states.shape[-2]] = key_states 
        self.key_cache[layer_idx][:, self.cachelength:self.cachelength + key_states.shape[-3]] = key_states 
        # self.value_cache[layer_idx][:, :, self.cachelength:self.cachelength + value_states.shape[-2]] = value_states 
        self.value_cache[layer_idx][:, self.cachelength:self.cachelength + value_states.shape[-3]] = value_states 
        
        k_out = self.key_cache[layer_idx][:, : self.cachelength + key_states.shape[-3]] 
        v_out = self.value_cache[layer_idx][:, : self.cachelength + value_states.shape[-3]] 
        
        if layer_idx == self.num_layers - 1: 
            self.cachelength += key_states.shape[-3] 

        # k_out[:, :, cache_position] = key_states 
        # v_out[:, :, cache_position] = value_states 

        return k_out, v_out

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states that were seen by the model."""
        # Occupied cache == any slot in the 3rd dim (sequence length) holds a non-zero value. To save on compute, let's
        # limit the check to the first batch member and head dimension.
        # TODO: deprecate this function in favor of `cache_position`
        # assert (self.key_cache[layer_idx][0, 0].any(dim=-1)).sum() == self.cachelength 
        # return (self.key_cache[layer_idx][0].any(dim=-1)).sum() 
        return self.cachelength 

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states."""
        return self.max_cache_len

    def reset(self):
        """Resets the cache values while preserving the objects"""
        self.cachelength = 0 
        for layer_idx in range(len(self.key_cache)):
            # In-place ops prevent breaking the static address
            self.key_cache[layer_idx].zero_()
            self.value_cache[layer_idx].zero_() 
    
    def rollbackone(self): 
        self.cachelength -= 1 

class StaticCacheSDPA(Cache):
    """
    Static Cache class to be used with `torch.compile(model)`.

    Parameters:
        config (`PretrainedConfig):
            The configuration file defining the shape-related attributes required to initialize the static cache.
        max_batch_size (`int`):
            The maximum batch size with which the model will be used.
        max_cache_len (`int`):
            The maximum sequence length with which the model will be used.
        device (`torch.device`):
            The device on which the cache should be initialized. Should be the same as the layer.
        dtype (*optional*, defaults to `torch.float32`):
            The default `dtype` to use when initializing the layer.
    """

    def __init__(self, config, max_batch_size: int, max_cache_len: int, device, dtype=None): 
        super().__init__()
        self.max_batch_size = max_batch_size
        self.max_cache_len = config.max_position_embeddings if max_cache_len is None else max_cache_len
        # Some model define a custom `head_dim` != config.hidden_size // config.num_attention_heads
        self.head_dim = (
            config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
        )

        self.dtype = dtype if dtype is not None else torch.float32
        self.num_key_value_heads = (
            config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        )

        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        cache_shape = (max_batch_size, self.num_key_value_heads, self.max_cache_len, self.head_dim)
        for _ in range(config.num_hidden_layers):
            # Note: `mark_static_address` is used to tag the cache as an fixed data pointer, preventing cuda graph
            # breaks when updating the cache.
            new_layer_key_cache = torch.zeros(cache_shape, dtype=self.dtype, device=device)
            new_layer_value_cache = torch.zeros(cache_shape, dtype=self.dtype, device=device)
            torch._dynamo.mark_static_address(new_layer_key_cache)
            torch._dynamo.mark_static_address(new_layer_value_cache)
            self.key_cache.append(new_layer_key_cache)
            self.value_cache.append(new_layer_value_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        It is VERY important to index using a tensor, otherwise you introduce a copy to the device.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The `StaticCache` needs the `cache_position` input
                to know how where to write in the cache.

        Return:
            A tuple containing the updated key and value states.
        """
        cache_position = cache_kwargs.get("cache_position")
        k_out = self.key_cache[layer_idx]
        v_out = self.value_cache[layer_idx]

        k_out[:, :, cache_position] = key_states
        v_out[:, :, cache_position] = value_states

        return k_out, v_out

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states that were seen by the model."""
        # Occupied cache == any slot in the 3rd dim (sequence length) holds a non-zero value. To save on compute, let's
        # limit the check to the first batch member and head dimension.
        # TODO: deprecate this function in favor of `cache_position`
        return (self.key_cache[layer_idx][0, 0].any(dim=-1)).sum()

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states."""
        return self.max_cache_len

    def reset(self):
        """Resets the cache values while preserving the objects"""
        for layer_idx in range(len(self.key_cache)):
            # In-place ops prevent breaking the static address
            self.key_cache[layer_idx].zero_()
            self.value_cache[layer_idx].zero_() 
    
    def backgather(self, indices, start, ends): 
        for layer_idx in range(len(self.key_cache)): 
            self.key_cache[layer_idx][:, :, start : ends, :] = self.key_cache[layer_idx][:, :, indices, :] 
            self.value_cache[layer_idx][:, :, start : ends, :] = self.value_cache[layer_idx][:, :, indices, :] 

class StaticCache2(Cache): 
    """
    Static Cache class to be used with `torch.compile(model)`.

    Parameters:
        config (`PretrainedConfig):
            The configuration file defining the shape-related attributes required to initialize the static cache.
        max_batch_size (`int`):
            The maximum batch size with which the model will be used.
        max_cache_len (`int`):
            The maximum sequence length with which the model will be used.
        device (`torch.device`):
            The device on which the cache should be initialized. Should be the same as the layer.
        dtype (*optional*, defaults to `torch.float32`):
            The default `dtype` to use when initializing the layer.
    """

    def __init__(self, config, max_batch_size: int, max_cache_len: int, device, dtype=None): 
        super().__init__()
        self.max_batch_size = max_batch_size
        self.max_cache_len = config.max_position_embeddings if max_cache_len is None else max_cache_len
        # Some model define a custom `head_dim` != config.hidden_size // config.num_attention_heads
        self.head_dim = (
            config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
        )

        self.dtype = dtype if dtype is not None else torch.float32
        self.num_key_value_heads = (
            config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        )
        '''
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        cache_shape = (max_batch_size, self.num_key_value_heads, self.max_cache_len, self.head_dim)
        for _ in range(config.num_hidden_layers):
            # Note: `mark_static_address` is used to tag the cache as an fixed data pointer, preventing cuda graph
            # breaks when updating the cache.
            new_layer_key_cache = torch.zeros(cache_shape, dtype=self.dtype, device=device)
            new_layer_value_cache = torch.zeros(cache_shape, dtype=self.dtype, device=device)
            torch._dynamo.mark_static_address(new_layer_key_cache)
            torch._dynamo.mark_static_address(new_layer_value_cache)
            self.key_cache.append(new_layer_key_cache)
            self.value_cache.append(new_layer_value_cache) 
        ''' 
        self.key_cache = torch.zeros((config.num_hidden_layers, max_batch_size, self.num_key_value_heads, self.max_cache_len, self.head_dim), dtype = self.dtype, device = device) 
        self.value_cache = torch.zeros((config.num_hidden_layers, max_batch_size, self.num_key_value_heads, self.max_cache_len, self.head_dim), dtype = self.dtype, device = device) 
        
        # self.key_cache = self.key_cache.contiguous() 
        # self.value_cache = self.value_cache.contiguous() 
        for layer_idx in range(config.num_hidden_layers): 
            torch._dynamo.mark_static_address(self.key_cache[layer_idx]) 
            torch._dynamo.mark_static_address(self.value_cache[layer_idx]) 
        torch._dynamo.mark_static_address(self.key_cache) 
        torch._dynamo.mark_static_address(self.value_cache) 

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        It is VERY important to index using a tensor, otherwise you introduce a copy to the device.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The `StaticCache` needs the `cache_position` input
                to know how where to write in the cache.

        Return:
            A tuple containing the updated key and value states.
        """
        cache_position = cache_kwargs.get("cache_position")
        k_out = self.key_cache[layer_idx] 
        v_out = self.value_cache[layer_idx] 

        k_out[:, :, cache_position] = key_states 
        v_out[:, :, cache_position] = value_states 
        # self.key_cache[layer_idx][:, :, cache_position] = key_states 
        # self.value_cache[layer_idx][:, :, cache_position] = value_states 

        # return self.key_cache[layer_idx], self.value_cache[layer_idx] 
        return k_out.clone(), v_out.clone() 

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states that were seen by the model."""
        # Occupied cache == any slot in the 3rd dim (sequence length) holds a non-zero value. To save on compute, let's
        # limit the check to the first batch member and head dimension.
        # TODO: deprecate this function in favor of `cache_position`
        return (self.key_cache[layer_idx, 0, 0].any(dim = -1)).sum() 

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states."""
        return self.max_cache_len

    def reset(self):
        """Resets the cache values while preserving the objects"""
        # for layer_idx in range(len(self.key_cache)): 
            # In-place ops prevent breaking the static address
            # self.key_cache[layer_idx].zero_() 
            # self.value_cache[layer_idx].zero_() 
        self.key_cache.zero_() 
        self.value_cache.zero_() 
    
    def backgather(self, indices, start, ends): 
        # for layer_idx in range(len(self.key_cache)): 
        #     self.key_cache[layer_idx][:, :, start : ends, :] = self.key_cache[layer_idx][:, :, indices, :] 
        #     self.value_cache[layer_idx][:, :, start : ends, :] = self.value_cache[layer_idx][:, :, indices, :] 
        self.key_cache[:, :, :, start : ends, :] = self.key_cache[:, :, :, indices, :] 
        self.value_cache[:, :, :, start : ends, :] = self.value_cache[:, :, :, indices, :] 
