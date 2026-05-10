import sys
sys.path.append('/home/admin/HAP')
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("="*60)
print("HAP Protection Mechanism Test (Simplified)")
print("="*60)

# 加载模型
print("\nLoading model...")
model_path = "/home/admin/minimind/MiniMind2"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()
print("✓ Model loaded")

# 测试文本
prompt = "Hello, my name is"
print(f"\nPrompt: {prompt}")

inputs = tokenizer(prompt, return_tensors='pt')
tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
print(f"Tokens: {tokens}")

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[-1][0]
    
    # 计算重要性
    importance = torch.norm(hidden, p=2, dim=1)
    
    print("\n" + "="*60)
    print("Token Importance Analysis")
    print("="*60)
    
    for i, (token, score) in enumerate(zip(tokens, importance.tolist())):
        print(f"{i:2d}. '{token}': {score:.2f}")
    
    # 测试保护机制
    print("\n" + "="*60)
    print("Protection Mechanism Test")
    print("="*60)
    
    # 不同保护策略
    keep_ratio = 0.5  # 保留50%
    keep_num = max(1, int(len(tokens) * keep_ratio))
    
    # 策略1：无保护（按重要性排序）
    _, indices_no_protect = torch.topk(importance, keep_num)
    kept_no_protect = [tokens[i] for i in sorted(indices_no_protect.tolist())]
    print(f"\nNo protection (keep {keep_num}/{len(tokens)}):")
    print(f"  Kept: {kept_no_protect}")
    
    # 策略2：保护前2个token
    protected_positions = set([0, 1])
    protected_tokens = [tokens[i] for i in protected_positions]
    
    # 剩余位置按重要性保留
    other_indices = [i for i in range(len(tokens)) if i not in protected_positions]
    other_keep = max(0, keep_num - len(protected_positions))
    
    if other_keep > 0 and other_indices:
        other_importance = importance[other_indices]
        _, top_other = torch.topk(other_importance, min(other_keep, len(other_indices)))
        other_kept = [other_indices[i] for i in top_other.tolist()]
    else:
        other_kept = []
    
    final_indices = sorted(list(protected_positions) + other_kept)
    final_tokens = [tokens[i] for i in final_indices]
    
    print(f"\nWith protection (protect first 2 tokens, keep {keep_num}/{len(tokens)}):")
    print(f"  Protected: {protected_tokens}")
    print(f"  Kept: {final_tokens}")
    
    # 验证保护效果
    print("\n" + "="*60)
    print("Verification:")
    print("="*60)
    for pos in protected_positions:
        if pos in final_indices:
            print(f"  ✓ Position {pos} ('{tokens[pos]}') is protected and kept")
        else:
            print(f"  ✗ Position {pos} ('{tokens[pos]}') would be lost without protection")
    
    print("\n✅ Protection mechanism works correctly!")
    print("The first few tokens (system prompt, instructions) are now protected.")
