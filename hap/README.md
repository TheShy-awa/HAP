# HAP: Hidden-state Aware Pruning for KV Cache

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

Efficient KV Cache pruning based on hidden state importance.  
**HAP** dynamically prunes KV cache during LLM inference, achieving high compression while preserving semantic meaning.

## 🔥 Core Idea

During LLM inference, hidden states of different tokens show significant variance in L2 norm:

| Token | L2 Norm | Importance |
|-------|---------|------------|
| "AI" (人工智能) | 46.16 | High → Keep |
| "is" (是) | 42.47 | Medium → Keep |
| "the" (的) | 40.88 | Low → Prune |

**Key insight**: Tokens with higher L2 norm carry more semantic information and should be retained.

## 📊 Results

Tested on `MiniMind2` with prompt: *"人工智能 是 未来 科技 发展 的 重要 方向"* (14 tokens)

| Compression | Kept | Kept Tokens |
|-------------|------|--------------|
| 71% | 4/14 | 人工智能, 是 |
| 50% | 7/14 | 人工智能, 是, 方向 |
| 36% | 9/14 | 人工智能, 是, 重要, 方向 |

> ✅ Semantic integrity preserved even at 71% compression!

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/HAP.git
cd HAP
pip install -r requirements.txt
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
