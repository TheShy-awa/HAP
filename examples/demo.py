#!/usr/bin/env python3
"""
Demo script for HAP (Hidden-state Aware Pruning) for KV Cache
"""

import sys
sys.path.append('..')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from hap import HAPPruner

def main():
    print("="*60)
    print("HAP Demo: Hidden-state Aware Pruning for KV Cache")
    print("="*60)
    
    # Load model (update this path to your model location)
    model_path = "/home/admin/minimind/MiniMind2"
    
    print(f"\nLoading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.tokenizer = tokenizer
    model.eval()
    print("✓ Model loaded successfully")
    
    # Initialize HAP pruner
    pruner = HAPPruner(model, keep_ratio=0.5, prune_interval=5)
    
    # Test prompt
    prompt = "Artificial intelligence is the future"
    print(f"\nPrompt: {prompt}")
    print("\nGenerating with HAP optimization...")
    
    output = pruner.generate(prompt, max_new_tokens=30)
    print(f"\nOutput: {output}")
    print("\n✅ Demo completed successfully!")

if __name__ == "__main__":
    main()
