import torch

class HAPPruner:
    """
    HAP (Hidden-state Aware Pruning) for KV Cache
    Compatible version for MiniMind and similar models
    """
    def __init__(self, model, keep_ratio=0.5, prune_interval=5):
        self.model = model
        self.keep_ratio = keep_ratio
        self.prune_interval = prune_interval
        self.protected_positions = [0, 1]  # Protect first 2 tokens

    def compute_importance(self, hidden_states):
        """Compute importance score for each token using L2 norm"""
        if len(hidden_states.shape) == 3:
            hidden_states = hidden_states[0]
        importance = torch.norm(hidden_states, p=2, dim=1)
        return importance

    def prune_kv_cache(self, past_key_values, importance, input_ids=None):
        """
        Prune KV cache with protection for important tokens
        """
        if past_key_values is None:
            return None

        seq_len = importance.shape[0]
        keep_num = max(1, int(seq_len * self.keep_ratio))
        
        # Determine protected indices
        protected = set()
        for pos in self.protected_positions:
            if pos < seq_len:
                protected.add(pos)
        
        # Calculate how many to keep beyond protected
        protected_num = len(protected)
        keep_regular = max(0, keep_num - protected_num)
        
        # Get regular indices (non-protected)
        regular_indices = [i for i in range(seq_len) if i not in protected]
        
        if keep_regular > 0 and len(regular_indices) > 0:
            regular_importance = importance[regular_indices]
            _, top_local = torch.topk(regular_importance, min(keep_regular, len(regular_indices)))
            regular_kept = [regular_indices[i] for i in top_local.tolist()]
        else:
            regular_kept = []
        
        # Combine protected + important regular tokens
        final_indices = sorted(list(protected) + regular_kept)
        
        # Prune each layer's KV cache
        new_past = []
        for layer_kv in past_key_values:
            # Handle different KV cache formats
            if isinstance(layer_kv, (tuple, list)):
                if len(layer_kv) >= 2:
                    k, v = layer_kv[0], layer_kv[1]
                else:
                    k = v = layer_kv[0]
            else:
                k = v = layer_kv
            
            # Prune along sequence dimension (dim=1)
            if len(k.shape) >= 3:
                new_k = k[:, final_indices, :, :] if k.shape[1] == seq_len else k
                new_v = v[:, final_indices, :, :] if v.shape[1] == seq_len else v
            else:
                new_k, new_v = k, v
            
            new_past.append((new_k, new_v))
        
        return tuple(new_past)

    def generate(self, prompt, max_new_tokens=50):
        """Generate text with HAP pruning enabled"""
        inputs = self.model.tokenizer(prompt, return_tensors='pt')
        input_ids = inputs.input_ids
        generated_ids = input_ids.clone()
        past_key_values = None
        step = 0

        print(f"Prompt: {prompt}")
        print(f"Input tokens: {input_ids.shape[1]}")

        with torch.no_grad():
            while step < max_new_tokens:
                # Forward pass
                if past_key_values is None:
                    outputs = self.model(
                        input_ids=input_ids,
                        use_cache=True,
                        output_hidden_states=True
                    )
                else:
                    outputs = self.model(
                        input_ids=generated_ids[:, -1:],
                        past_key_values=past_key_values,
                        use_cache=True,
                        output_hidden_states=True
                    )

                # Get next token
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                past_key_values = outputs.past_key_values

                # Apply HAP pruning periodically
                if step > 0 and step % self.prune_interval == 0 and outputs.hidden_states is not None:
                    hidden = outputs.hidden_states[-1]
                    if len(hidden.shape) == 3:
                        hidden = hidden[0]
                    importance = self.compute_importance(hidden)
                    past_key_values = self.prune_kv_cache(
                        past_key_values, importance, input_ids=generated_ids
                    )

                step += 1
                
                # Stop if EOS
                if next_token.item() == self.model.tokenizer.eos_token_id:
                    break

            return self.model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    def set_protected_positions(self, positions):
        """Set which positions to protect (e.g., [0,1] for first two tokens)"""
        self.protected_positions = positions
