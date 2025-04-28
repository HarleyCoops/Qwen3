# Qwen3-235B-A22B Implementation Guide

This guide provides practical implementations and code examples for working with the Qwen3-235B-A22B model. These examples can be converted to a Jupyter notebook format for interactive exploration.

## 1. Environment Setup

First, let's set up our environment with the necessary dependencies:

```python
# Install required packages
!pip install transformers>=4.51.0 torch>=2.6.0
```

### Hardware Requirements

The Qwen3-235B-A22B model is a large MoE model with 235B total parameters (22B activated). Here are the minimum hardware requirements:

* For full precision (BF16): Multiple high-end GPUs with at least 80GB+ VRAM and NVLink
* For quantized versions: A40/A100/H100 GPUs or similar
* For smaller devices: Consider using a smaller model in the Qwen3 family or quantized versions

## 2. Basic Usage with Transformers

### 2.1 Loading the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-235B-A22B"

# For production use, specify appropriate torch_dtype and device_map
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"  # This will distribute the model across available GPUs
)
```

### 2.2 Basic Inference with Thinking Mode (Default)

```python
# Prepare input
prompt = "Explain the concept of Mixture of Experts in large language models."
messages = [{"role": "user", "content": prompt}]

# Format with chat template (thinking mode enabled by default)
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True  # Default is True
)

# Tokenize and generate
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768,  # Recommended length for complex responses
    temperature=0.6,       # Recommended for thinking mode
    top_p=0.95,
    top_k=20,
    min_p=0
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

# Extract thinking content and final response
try:
    # Find </think> token (151668)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
final_response = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("THINKING PROCESS:")
print(thinking_content)
print("\nFINAL RESPONSE:")
print(final_response)
```

### 2.3 Using Non-Thinking Mode

```python
# Format with chat template (thinking mode disabled)
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False  # Explicitly disable thinking
)

# Tokenize and generate
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=8192,  # Can use shorter length for non-thinking mode
    temperature=0.7,      # Recommended for non-thinking mode
    top_p=0.8,
    top_k=20,
    min_p=0
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
response = tokenizer.decode(output_ids, skip_special_tokens=True)

print("RESPONSE (Non-thinking mode):")
print(response)
```

### 2.4 Using Soft Switches for Thinking Control

```python
# Example of using soft switch in the prompt
think_prompt = "Solve this complex math problem: If a train travels at 60 mph and another at 75 mph in opposite directions, how long will it take for them to be 270 miles apart if they start at the same location? /think"
no_think_prompt = "Tell me a short joke about programming. /no_think"

# Process both prompts
for prompt, mode in [(think_prompt, "thinking"), (no_think_prompt, "non-thinking")]:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True  # We use soft switches in the prompt
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    
    # Extract thinking content and final response
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
    
    thinking = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    response = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    
    print(f"\n--- PROMPT WITH {mode.upper()} MODE ---")
    print(f"Prompt: {prompt}")
    print(f"Thinking: {thinking if thinking else 'No thinking content generated'}")
    print(f"Response: {response}")
```

## 3. Multi-Turn Conversations

```python
class Qwen3Chatbot:
    def __init__(self, model_name="Qwen/Qwen3-235B-A22B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.history = []
        self.enable_thinking = True
    
    def generate_response(self, user_input, parse_thinking=True):
        messages = self.history + [{"role": "user", "content": user_input}]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        # Use appropriate parameters based on thinking mode
        if self.enable_thinking:
            temp, top_p = 0.6, 0.95
        else:
            temp, top_p = 0.7, 0.8
            
        output_ids = self.model.generate(
            **inputs, 
            max_new_tokens=32768,
            temperature=temp,
            top_p=top_p,
            top_k=20,
            min_p=0
        )[0][len(inputs.input_ids[0]):].tolist()
        
        if parse_thinking and self.enable_thinking:
            try:
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0
                
            thinking = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            response = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            
            # Update history with only the final response
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": response})
            
            return {"thinking": thinking, "response": response}
        else:
            response = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
            
            # Update history
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": response})
            
            return {"thinking": None, "response": response}
    
    def toggle_thinking_mode(self, enable=None):
        if enable is not None:
            self.enable_thinking = enable
        else:
            self.enable_thinking = not self.enable_thinking
        return f"Thinking mode {'enabled' if self.enable_thinking else 'disabled'}"

# Example usage
chatbot = Qwen3Chatbot()

# First message with thinking enabled
result1 = chatbot.generate_response("What are three possible applications of MoE models?")
print(f"Thinking: {result1['thinking']}")
print(f"Response: {result1['response']}")

# Follow up question continuing the conversation
result2 = chatbot.generate_response("Can you elaborate on the second application?")
print(f"Thinking: {result2['thinking']}")
print(f"Response: {result2['response']}")

# Toggle thinking mode off
chatbot.toggle_thinking_mode(False)
result3 = chatbot.generate_response("Thanks! Now summarize our conversation briefly.")
print(f"Response: {result3['response']}")
```

## 4. Advanced Features

### 4.1 Long Context Processing with YaRN

```python
# Option 1: Using transformers with YaRN configuration
from transformers import AutoModelForCausalLM, AutoTokenizer

# Setup model with YaRN scaling
yarn_config = {
    "rope_scaling": {
        "type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 32768
    }
}

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-235B-A22B",
    torch_dtype="auto",
    device_map="auto",
    **yarn_config  # Apply YaRN scaling
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-235B-A22B")

# Now the model can handle up to 131,072 tokens

# Option 2: When using vLLM
# vllm serve Qwen/Qwen3-235B-A22B ... --rope-scaling '{"type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' --max-model-len 131072

# Option 3: When using SGLang
# python -m sglang.launch_server ... --json-model-override-args '{"rope_scaling":{"type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}'
```

### 4.2 Tool Use and Agent Capabilities

```python
# Using Qwen-Agent is recommended for tool integration
# Installation: pip install qwen-agent

from qwen_agent.agents import Assistant

# Define LLM configuration
llm_cfg = {
    'model': 'Qwen3-235B-A22B',
    
    # Use custom endpoint compatible with OpenAI API
    'model_server': 'http://localhost:8000/v1',  # api_base
    'api_key': 'EMPTY',
}

# Define tools using MCP configuration
tools = [
    {'mcpServers': {  # MCP configuration
            'time': {
                'command': 'uvx',
                'args': ['mcp-server-time', '--local-timezone=Asia/Shanghai']
            },
            "fetch": {
                "command": "uvx",
                "args": ["mcp-server-fetch"]
            }
        }
    },
    'code_interpreter',  # Built-in tool
]

# Initialize agent
agent = Assistant(llm=llm_cfg, function_list=tools)

# Run the agent
messages = [{'role': 'user', 'content': 'What time is it now and what are the main features of Qwen3?'}]
for responses in agent.run(messages=messages):
    pass  # In actual usage, you might want to process intermediate responses

print(responses)
```

### 4.3 Deployment with vLLM for Production

```bash
# Install vLLM: pip install vllm>=0.8.4

# Start server with reasoning capability enabled
vllm serve Qwen/Qwen3-235B-A22B --port 8000 --enable-reasoning --reasoning-parser deepseek_r1

# For production deployment with thinking mode disabled by default
# vllm serve Qwen/Qwen3-235B-A22B --port 8000 --chat-template-kwargs '{"enable_thinking": false}'
```

Python client to interact with vLLM server:

```python
from openai import OpenAI

# Configure client
client = OpenAI(
    api_key="EMPTY",  # Not needed for local deployment
    base_url="http://localhost:8000/v1"
)

# Example with thinking mode
response = client.chat.completions.create(
    model="Qwen/Qwen3-235B-A22B",
    messages=[
        {"role": "user", "content": "Explain the advantage of MoE architecture over dense models"}
    ],
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    max_tokens=32768
)

print(response.choices[0].message.content)

# Example with thinking disabled
response_no_thinking = client.chat.completions.create(
    model="Qwen/Qwen3-235B-A22B",
    messages=[
        {"role": "user", "content": "Give a one-paragraph summary of MoE architecture"}
    ],
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    presence_penalty=1.5,
    max_tokens=8192,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}}
)

print(response_no_thinking.choices[0].message.content)
```

## 5. Best Practices

### 5.1 Parameter Settings

For optimal performance with Qwen3-235B-A22B, use these recommended settings:

**Thinking Mode (enable_thinking=True)**:
- Temperature: 0.6
- Top-p: 0.95
- Top-k: 20
- Min-p: 0
- Max tokens: 32,768 (or higher for complex problems)
- Do NOT use greedy decoding (temperature=0)

**Non-Thinking Mode (enable_thinking=False)**:
- Temperature: 0.7
- Top-p: 0.8
- Top-k: 20
- Min-p: 0
- Max tokens: 8,192 (can be lower for simple responses)
- For reducing repetition: Add presence_penalty between 0-2

### 5.2 Standardizing Output Format

For consistent outputs, include formatting instructions in your prompts:

**Math Problems**:
- Add "Please reason step by step, and put your final answer within \\boxed{}." to your prompt

**Multiple-Choice Questions**:
- Add "Please show your choice in the `answer` field with only the choice letter, e.g., `"answer": "C"`." to your prompt

### 5.3 Multi-Turn Conversation Best Practices

- Do not include thinking content in conversation history
- Only store the final responses in history
- Update the latest thinking switch in multi-turn conversations

## 6. Lightweight Alternatives

For resource-constrained environments, consider these alternatives:

### 6.1 Using Smaller Models in the Qwen3 Family

```python
# Try a smaller model from the Qwen3 family
smaller_model = "Qwen/Qwen3-8B"  # Only 8B parameters

tokenizer = AutoTokenizer.from_pretrained(smaller_model)
model = AutoModelForCausalLM.from_pretrained(
    smaller_model,
    torch_dtype="auto",
    device_map="auto"
)
```

### 6.2 Using Quantized Versions

```python
# Using 4-bit quantized version
quantized_model = "Qwen/Qwen3-235B-A22B-Q4_0"

tokenizer = AutoTokenizer.from_pretrained(quantized_model)
model = AutoModelForCausalLM.from_pretrained(
    quantized_model,
    device_map="auto"
)
```

### 6.3 Using Ollama for Local Deployment

```bash
# Install Ollama from https://ollama.com/
# Start Ollama service
ollama serve

# Pull the model (use a smaller variant if needed)
ollama run qwen3:8b
```

## Conclusion

This implementation guide demonstrates key patterns and practices for working with the Qwen3-235B-A22B model. By following these examples, you can effectively leverage the model's capabilities in both thinking and non-thinking modes, while ensuring optimal performance across various deployment scenarios.
