import torch

class HAPPruner:
    """
    HAP (Hidden-state Aware Pruning) for KV Cache
    Dynamically prunes KV cache based on hidden state importance.
    """
    def __init__(self, model, keep_ratio=0.5, prune_interval=5):
        self.model = model
        self.keep_ratio = keep_ratio
        self.prune_interval = prune_interval

    def compute_importance(self, hidden_states):
        """Compute importance score for each token using L2 norm"""
        if len(hidden_states.shape) == 3:
            hidden_states = hidden_states[0]
        importance = torch.norm(hidden_states, p=2, dim=1)
        return importance

    def prune_kv_cache(self, past_key_values, importance):
        """Prune KV cache, keeping only top-k important tokens"""
        if past_key_values is None:
            return None

        seq_len = importance.shape[0]
        keep_num = max(1, int(seq_len * self.keep_ratio))

        _, top_indices = torch.topk(importance, keep_num)
        top_indices, _ = torch.sort(top_indices)

        new_past = []
        for k, v in past_key_values:
            new_k = k[:, top_indices, :, :]
            new_v = v[:, top_indices, :, :]
            new_past.append((new_k, new_v))

        return tuple(new_past)

    def generate(self, prompt, max_new_tokens=50):
        """Generate text with HAP pruning enabled"""
        inputs = self.model.tokenizer(prompt, return_tensors='pt')
        input_ids = inputs.input_ids
        generated_ids = input_ids.clone()
        past_key_values = None
        step = 0

        with torch.no_grad():
            while step < max_new_tokens:
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

                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                past_key_values = outputs.past_key_values

                if step > 0 and step % self.prune_interval == 0 and outputs.hidden_states is not None:
                    hidden = outputs.hidden_states[-1]
                    if len(hidden.shape) == 3:
                        hidden = hidden[0]
                    importance = self.compute_importance(hidden)
                    past_key_values = self.prune_kv_cache(past_key_values, importance)

                step += 1

            return self.model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
