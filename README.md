# Qwen3-235B-A22B Educational Project

## Overview

This educational project explores the Qwen3-235B-A22B model - a state-of-the-art Mixture-of-Experts (MoE) language model with 235B total parameters and 22B active parameters. The project aims to comprehensively understand the model's capabilities, architecture, training methodology, and optimal usage patterns through minimal code changes while leveraging official resources.

## Project Resources

This repository contains the following key resources:

### 1. [Project Plan](qwen3_educational_project_plan.md)

A comprehensive plan for exploring and documenting Qwen3-235B-A22B, including:
- Model architecture and specifications
- Research methodologies
- Implementation plans
- Educational content development strategies
- Timeline and milestones

### 2. [Implementation Guide](qwen3_implementation_guide.ipynb.md)

A practical guide with code examples demonstrating how to:
- Set up the environment and necessary dependencies
- Use the model with both thinking and non-thinking modes
- Build multi-turn conversations
- Leverage the model's long-context capabilities
- Deploy using various frameworks (Transformers, vLLM, SGLang)
- Integrate tool use and agent capabilities

### 3. [Benchmarking and Evaluation Guide](qwen3_benchmark_evaluation_guide.md)

A systematic approach to evaluating the model's performance:
- Comprehensive evaluation framework across key capability areas
- Benchmark datasets and tasks for different domains
- Comparative evaluation between thinking and non-thinking modes
- YaRN long-context performance assessment
- Multilingual capability benchmarking
- Tool use and agent capability evaluation

## Key Model Features

The Qwen3-235B-A22B model offers several groundbreaking capabilities:

1. **Dual-Mode Operation**: Unique support for seamless switching between thinking mode (for complex logical reasoning, math, and coding) and non-thinking mode (for efficient, general-purpose dialogue) within a single model.

2. **Enhanced Reasoning**: Significantly improved reasoning capabilities in mathematics, code generation, and commonsense logical reasoning.

3. **Human Preference Alignment**: Superior alignment with human preferences for creative writing, role-playing, multi-turn dialogues, and instruction following.

4. **Agent Capabilities**: Expert integration with external tools in both thinking and unthinking modes.

5. **Multilingual Support**: Strong capabilities across 100+ languages and dialects with robust instruction following and translation abilities.

6. **Extended Context Length**: Native support for 32,768 tokens, extendable to 131,072 tokens using YaRN.

## Getting Started

To begin exploring the Qwen3-235B-A22B model:

1. Review the [Project Plan](qwen3_educational_project_plan.md) to understand the educational approach.
2. Follow the [Implementation Guide](qwen3_implementation_guide.ipynb.md) to set up and start using the model.
3. Use the [Benchmarking and Evaluation Guide](qwen3_benchmark_evaluation_guide.md) to systematically assess the model's capabilities.

## Hardware Requirements

The Qwen3-235B-A22B model requires significant computational resources:

- For full precision (BF16): Multiple high-end GPUs with at least 80GB+ VRAM and NVLink
- For quantized versions: A40/A100/H100 GPUs or similar
- For resource-constrained environments: Consider smaller models in the Qwen3 family or quantized versions

## Official Resources

This project leverages the following official resources:

- [Hugging Face Model Page](https://huggingface.co/Qwen/Qwen3-235B-A22B)
- [GitHub Repository](https://github.com/QwenLM/Qwen3)
- [Official Documentation](https://qwen.readthedocs.io/)
- [Official Blog](https://qwenlm.github.io/blog/qwen3/)

## Best Practices

### Thinking Mode (enable_thinking=True)
- Temperature: 0.6
- Top-p: 0.95
- Top-k: 20
- Min-p: 0
- DO NOT use greedy decoding (avoid temperature=0)

### Non-Thinking Mode (enable_thinking=False)
- Temperature: 0.7
- Top-p: 0.8
- Top-k: 20
- Min-p: 0
- For reducing repetition: presence_penalty between 0-2

## Contribution

This educational project is designed to evolve with community contributions and insights. Feel free to submit pull requests with additional examples, benchmarks, or documentation improvements.

## License

This educational project follows the same licensing as the Qwen3 model itself, which is Apache 2.0 licensed.
