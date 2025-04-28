# Qwen3-235B-A22B Benchmarking and Evaluation Guide

This guide outlines a comprehensive approach to benchmarking and evaluating the Qwen3-235B-A22B model across different domains and capabilities. It specifically focuses on evaluating the model's performance in both thinking and non-thinking modes, which is a unique feature of the Qwen3 series.

## 1. Evaluation Framework

### 1.1 Key Performance Areas

When evaluating Qwen3-235B-A22B, focus on these key capabilities:

1. **Reasoning and Problem-Solving**
   - Mathematical reasoning
   - Logical deduction
   - Complex multi-step reasoning

2. **Knowledge and Information Processing**
   - Factual accuracy
   - Knowledge breadth
   - Knowledge depth in specialized domains

3. **Language Understanding and Generation**
   - Natural language understanding
   - Creative writing
   - Content summarization
   - Translation quality

4. **Code Generation and Analysis**
   - Algorithm implementation
   - Code debugging
   - Code explanation
   - Language support breadth

5. **Agent and Tool Use Capabilities**
   - Task planning
   - Tool selection
   - Tool use execution
   - Feedback incorporation

6. **Multilingual Performance**
   - Non-English instruction following
   - Cross-lingual understanding
   - Translation quality

### 1.2 Evaluation Modes

For each performance area, evaluate the model in:

- **Thinking Mode** (enable_thinking=True, default)
- **Non-Thinking Mode** (enable_thinking=False)
- **Mixed Mode** (using /think and /no_think directives)

## 2. Benchmark Datasets and Tasks

### 2.1 Reasoning Benchmarks

| Benchmark | Description | Evaluation Focus |
|-----------|-------------|-----------------|
| GSM8K | Grade school math problems | Mathematical reasoning |
| MATH | College-level math problems | Advanced mathematical reasoning |
| BBH | Big-Bench Hard tasks | Diverse reasoning tasks |
| TheoremQA | Mathematical theorem application | Formal logical reasoning |
| LogiQA | Logical reasoning questions | Deductive reasoning |

**Sample Evaluation Code:**

```python
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re

model_name = "Qwen/Qwen3-235B-A22B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load evaluation dataset (GSM8K example)
with open("gsm8k_test.jsonl", "r") as f:
    examples = [json.loads(line) for line in f]

results = {
    "thinking_mode": {"correct": 0, "total": 0},
    "non_thinking_mode": {"correct": 0, "total": 0}
}

# Extract the numeric answer from the model output
def extract_answer(text):
    # Look for answer in boxed format first
    boxed_match = re.search(r"\\boxed\{(.*?)\}", text)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    # Look for "The answer is" pattern
    answer_match = re.search(r"The answer is[:\s]*([\d\.\-]+)", text)
    if answer_match:
        return answer_match.group(1).strip()
    
    # Look for the last number in the text
    numbers = re.findall(r"(\d+\.?\d*)", text)
    if numbers:
        return numbers[-1].strip()
    
    return None

# Evaluate with thinking mode
for i, example in enumerate(examples[:100]):  # Limit to 100 examples for testing
    question = example["question"]
    answer = example["answer"]
    
    # Try with thinking mode
    messages = [{"role": "user", "content": question + " Please reason step by step, and put your final answer within \\boxed{}."}]
    
    # With thinking mode
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32768,
            temperature=0.6,
            top_p=0.95,
            top_k=20
        )
    
    output_text = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    model_answer = extract_answer(output_text)
    
    is_correct = model_answer == answer
    results["thinking_mode"]["total"] += 1
    if is_correct:
        results["thinking_mode"]["correct"] += 1
    
    # Now with non-thinking mode
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=8192,
            temperature=0.7,
            top_p=0.8,
            top_k=20
        )
    
    output_text = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    model_answer = extract_answer(output_text)
    
    is_correct = model_answer == answer
    results["non_thinking_mode"]["total"] += 1
    if is_correct:
        results["non_thinking_mode"]["correct"] += 1
    
    # Print progress
    if (i + 1) % 10 == 0:
        print(f"Processed {i+1} examples")
        print(f"Thinking mode accuracy: {results['thinking_mode']['correct'] / results['thinking_mode']['total']:.2f}")
        print(f"Non-thinking mode accuracy: {results['non_thinking_mode']['correct'] / results['non_thinking_mode']['total']:.2f}")

# Print final results
print("Final results:")
print(f"Thinking mode accuracy: {results['thinking_mode']['correct'] / results['thinking_mode']['total']:.4f}")
print(f"Non-thinking mode accuracy: {results['non_thinking_mode']['correct'] / results['non_thinking_mode']['total']:.4f}")
```

### 2.2 Knowledge and Information Processing

| Benchmark | Description | Evaluation Focus |
|-----------|-------------|-----------------|
| MMLU | Massive Multitask Language Understanding | Breadth of knowledge across domains |
| TruthfulQA | Evaluate model honesty and accuracy | Factual accuracy |
| NaturalQuestions | Real user queries from Google Search | Information retrieval and synthesis |
| TriviaQA | Trivia questions across various domains | Broad knowledge assessment |
| BioASQ | Biomedical semantic QA challenge | Domain-specific knowledge (medical) |

### 2.3 Language Understanding and Generation

| Benchmark | Description | Evaluation Focus |
|-----------|-------------|-----------------|
| DROP | Reading comprehension requiring numerical reasoning | Complex language understanding |
| HellaSwag | Commonsense reasoning for text completion | Contextual understanding |
| WinoGrande | Commonsense reasoning | Pronoun resolution, context understanding |
| MT-Bench | Multi-turn conversation benchmark | Conversational abilities |
| HHEM | Human Preference Alignment | Creative writing, dialogue quality |

### 2.4 Code Generation and Analysis

| Benchmark | Description | Evaluation Focus |
|-----------|-------------|-----------------|
| HumanEval | Python code generation tasks | Functional correctness |
| MBPP | Python programming problems | Code generation quality |
| DS-1000 | Data science coding tasks | Domain-specific coding |
| LeetCode-Hard | Algorithmic problem solving | Complex algorithm implementation |
| CodeInterpreter-Hard | Code generation with explanations | Code + explanation quality |

### 2.5 Agent and Tool Use Capabilities

| Benchmark | Description | Evaluation Focus |
|-----------|-------------|-----------------|
| AgentBench | Multi-domain agent evaluation | Agent behavior quality |
| ToolBench | Tool selection and use | Appropriate tool selection |
| Mind2Web | Web agent evaluation | Web navigation and interaction |
| API-Bank | API usage evaluation | Correct API understanding and use |
| OS-Copilot-Benchmark | Operating system task automation | System command understanding |

### 2.6 Multilingual Performance

| Benchmark | Description | Evaluation Focus |
|-----------|-------------|-----------------|
| XNLI | Cross-lingual Natural Language Inference | Cross-lingual understanding |
| TyDi QA | Typologically Diverse Question Answering | Diverse language question answering |
| FLORES-200 | Machine translation benchmark | Translation quality |
| XStoryCloze | Multilingual commonsense reasoning | Reasoning across languages |
| XCOPA | Cross-lingual Choice of Plausible Alternatives | Causal reasoning in multiple languages |

## 3. Comparative Evaluation: Thinking vs. Non-Thinking Mode

### 3.1 Task Categories for Mode Comparison

| Task Category | Expected Better Mode | Rationale |
|---------------|----------------------|-----------|
| Complex mathematical reasoning | Thinking | Benefits from step-by-step reasoning |
| Code generation | Thinking | Benefits from algorithm planning |
| Simple fact retrieval | Non-thinking | Direct answer more efficient |
| Creative writing | Non-thinking | Fluent generation without over-analysis |
| Multi-step planning | Thinking | Benefits from breaking down complex tasks |
| Conversational responses | Non-thinking | More natural, less verbose |
| Language translation | Non-thinking | Direct mapping more efficient |
| Tool use reasoning | Thinking | Benefits from reasoning about tool selection |

### 3.2 Evaluation Methodology for Mode Comparison

For a rigorous comparison between thinking and non-thinking modes:

1. **Paired Evaluation**: Evaluate the same examples with both modes
2. **Multiple Metrics**: Measure:
   - Accuracy/correctness
   - Response generation time
   - Token efficiency (tokens used vs. information conveyed)
   - Hallucination frequency
   - Reasoning transparency
   - Confidence calibration

3. **Human Evaluation**: For subjective tasks, use human evaluators to rate:
   - Response quality
   - Helpfulness
   - Naturalness
   - Reasoning quality

### 3.3 Sample Evaluation Script for Mode Comparison

```python
import pandas as pd
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen3-235B-A22B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Comparison tasks with expected better mode
tasks = [
    {"category": "Mathematical Reasoning", "prompt": "Solve for x: 3x^2 + 6x - 9 = 0", "expected_better": "thinking"},
    {"category": "Code Generation", "prompt": "Write a Python function to find the nth Fibonacci number using dynamic programming", "expected_better": "thinking"},
    {"category": "Fact Retrieval", "prompt": "What is the capital of France?", "expected_better": "non-thinking"},
    {"category": "Creative Writing", "prompt": "Write a short poem about autumn", "expected_better": "non-thinking"},
    {"category": "Planning", "prompt": "Plan a 7-day itinerary for a trip to Japan", "expected_better": "thinking"},
    {"category": "Conversation", "prompt": "How's the weather today?", "expected_better": "non-thinking"},
    {"category": "Tool Use", "prompt": "I need to extract data from a CSV file and perform statistical analysis. Which tools should I use?", "expected_better": "thinking"},
]

results = []

for task in tasks:
    task_results = {
        "category": task["category"],
        "prompt": task["prompt"],
        "expected_better": task["expected_better"]
    }
    
    messages = [{"role": "user", "content": task["prompt"]}]
    
    # Evaluate with thinking mode
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # Measure generation time
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32768,
            temperature=0.6,
            top_p=0.95,
            top_k=20
        )
    thinking_time = time.time() - start_time
    
    output_text = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    thinking_tokens = len(outputs[0]) - len(inputs.input_ids[0])
    
    # Evaluate with non-thinking mode
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # Measure generation time
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=8192,
            temperature=0.7,
            top_p=0.8,
            top_k=20
        )
    non_thinking_time = time.time() - start_time
    
    output_text_non_thinking = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    non_thinking_tokens = len(outputs[0]) - len(inputs.input_ids[0])
    
    # Record results
    task_results.update({
        "thinking_time": thinking_time,
        "non_thinking_time": non_thinking_time,
        "thinking_tokens": thinking_tokens,
        "non_thinking_tokens": non_thinking_tokens,
        "time_ratio": thinking_time / non_thinking_time if non_thinking_time > 0 else float('inf'),
        "token_ratio": thinking_tokens / non_thinking_tokens if non_thinking_tokens > 0 else float('inf')
    })
    
    results.append(task_results)

# Convert to DataFrame for analysis
df = pd.DataFrame(results)
print(df)

# Group by category and compute averages
category_metrics = df.groupby('category').agg({
    'thinking_time': 'mean',
    'non_thinking_time': 'mean',
    'thinking_tokens': 'mean',
    'non_thinking_tokens': 'mean',
    'time_ratio': 'mean',
    'token_ratio': 'mean'
})

print("\nCategory Averages:")
print(category_metrics)

# Group by expected better mode and compute averages
mode_metrics = df.groupby('expected_better').agg({
    'thinking_time': 'mean',
    'non_thinking_time': 'mean',
    'thinking_tokens': 'mean',
    'non_thinking_tokens': 'mean',
    'time_ratio': 'mean',
    'token_ratio': 'mean'
})

print("\nExpected Better Mode Averages:")
print(mode_metrics)
```

## 4. Evaluating YaRN Long-Context Performance

### 4.1 Long-Context Tasks

| Task | Description | Context Length |
|------|-------------|---------------|
| Document QA | Answer questions about a long document | 50K-100K tokens |
| Multi-document synthesis | Synthesize information across multiple documents | 50K-100K tokens |
| Book summarization | Summarize an entire book | 100K+ tokens |
| Code base navigation | Answer questions about a large codebase | 50K-100K tokens |
| Long conversation tracking | Track details in multi-turn long conversations | 20K-40K tokens |

### 4.2 YaRN Configuration Testing

Test various YaRN factor settings to find optimal performance:

1. Default (no YaRN): Native 32,768 token context
2. YaRN factor 2.0: Extended to ~65,536 tokens
3. YaRN factor 4.0: Extended to ~131,072 tokens

### 4.3 Evaluation Metrics for Long-Context Tasks

- **Retrieval accuracy**: Ability to recall information from early parts of the context
- **Coherence**: Maintaining a coherent response across long inputs
- **Hallucination rate**: Tendency to hallucinate with increasing context length
- **Memory efficiency**: RAM/VRAM usage at different context lengths
- **Inference speed**: Generation time at different context lengths

## 5. Benchmark for Multilingual Capabilities

### 5.1 Languages to Test

Test a diverse set of languages across different language families:

1. **Indo-European**: English, Spanish, French, German, Russian, Hindi
2. **Sino-Tibetan**: Chinese (Simplified/Traditional), Tibetan
3. **Afro-Asiatic**: Arabic, Hebrew, Amharic
4. **Austronesian**: Indonesian, Tagalog
5. **Japonic**: Japanese
6. **Koreanic**: Korean
7. **Dravidian**: Tamil, Telugu
8. **Niger-Congo**: Swahili, Yoruba
9. **Turkic**: Turkish, Kazakh
10. **Other**: Thai, Vietnamese

### 5.2 Multilingual Tasks to Evaluate

For each language, evaluate:

1. **Instruction following**: Basic commands and instructions
2. **Text completion**: Complete partial text naturally
3. **Translation**: Translate to and from English
4. **Summarization**: Summarize lengthy content
5. **QA**: Answer questions in the target language
6. **Content creation**: Generate creative content

### 5.3 Example Multilingual Evaluation Script

```python
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen3-235B-A22B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Sample multilingual tasks
languages = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi"
}

# Task templates for each language
task_templates = {
    "instruction": {
        "en": "List five advantages of electric vehicles.",
        "es": "Enumera cinco ventajas de los vehículos eléctricos.",
        "fr": "Énumérez cinq avantages des véhicules électriques.",
        "de": "Nenne fünf Vorteile von Elektrofahrzeugen.",
        "ru": "Перечислите пять преимуществ электромобилей.",
        "zh": "列举电动汽车的五个优势。",
        "ja": "電気自動車の5つの利点を挙げてください。",
        "ko": "전기 자동차의 다섯 가지 장점을 나열하세요.",
        "ar": "عدد خمس مزايا للسيارات الكهربائية.",
        "hi": "इलेक्ट्रिक वाहनों के पांच फायदे बताएं।"
    },
    "translation": {
        "en": "Translate the following to French: 'Machine learning is transforming how we interact with technology.'",
        "es": "Traduce al inglés: 'El aprendizaje automático está transformando la forma en que interactuamos con la tecnología.'",
        "fr": "Traduisez en anglais: 'L'apprentissage automatique transforme notre façon d'interagir avec la technologie.'",
        "de": "Übersetze ins Englische: 'Maschinelles Lernen verändert die Art und Weise, wie wir mit Technologie interagieren.'",
        "ru": "Переведите на английский: 'Машинное обучение меняет то, как мы взаимодействуем с технологиями.'",
        "zh": "翻译成英文: '机器学习正在改变我们与技术互动的方式。'",
        "ja": "英語に翻訳してください: '機械学習は私たちが技術と対話する方法を変えています。'",
        "ko": "영어로 번역하세요: '기계 학습은 우리가 기술과 상호 작용하는 방식을 변화시키고 있습니다.'",
        "ar": "ترجم إلى الإنجليزية: 'التعلم الآلي يغير طريقة تفاعلنا مع التكنولوجيا.'",
        "hi": "अंग्रेजी में अनुवाद करें: 'मशीन लर्निंग बदल रही है कि हम तकनीक के साथ कैसे बातचीत करते हैं।'"
    },
    "qa": {
        "en": "What causes the seasons on Earth?",
        "es": "¿Qué causa las estaciones en la Tierra?",
        "fr": "Qu'est-ce qui cause les saisons sur Terre?",
        "de": "Was verursacht die Jahreszeiten auf der Erde?",
        "ru": "Что вызывает смену сезонов на Земле?",
        "zh": "是什么导致了地球上的季节变化？",
        "ja": "地球の季節は何が原因で起こりますか？",
        "ko": "지구에서 계절이 생기는 원인은 무엇인가요?",
        "ar": "ما الذي يسبب الفصول على الأرض؟",
        "hi": "पृथ्वी पर मौसम का कारण क्या है?"
    }
}

results = []

# Evaluate each language and task with both thinking and non-thinking modes
for task_name, task_by_lang in task_templates.items():
    for lang_code, prompt in task_by_lang.items():
        lang_name = languages[lang_code]
        
        # Test with thinking mode
        messages = [{"role": "user", "content": prompt}]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32768,
                temperature=0.6,
                top_p=0.95,
                top_k=20
            )
        
        output_text_thinking = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        
        # Test with non-thinking mode
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=8192,
                temperature=0.7,
                top_p=0.8,
                top_k=20
            )
        
        output_text_non_thinking = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        
        # Record result
        results.append({
            "task": task_name,
            "language": lang_name,
            "language_code": lang_code,
            "prompt": prompt,
            "thinking_response": output_text_thinking,
            "non_thinking_response": output_text_non_thinking
        })
        
        print(f"Completed {task_name} in {lang_name}")

# Save results to file
with open("multilingual_evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("Evaluation complete! Results saved to multilingual_evaluation_results.json")
```

## 6. Tool Use and Agent Capability Evaluation

### 6.1 Tool Use Evaluation Framework

Evaluate the model's capability to use tools effectively:

1. **Tool Selection**: Ability to choose the appropriate tool for a task
2. **Tool Use Accuracy**: Correct usage of the selected tool
3. **Parameter Handling**: Proper handling of tool parameters
4. **Integration**: Ability to integrate tool results into the response
5. **Error Handling**: Appropriate handling of tool errors or unexpected results

### 6.2 Agent Tasks to Evaluate

Create a set of increasingly complex agent tasks:

1. **Simple tool use**: Single tool usage for a specific task
2. **Multi-tool orchestration**: Using multiple tools in sequence
3. **Conditional tool use**: Selecting tools based on conditions
4. **Error recovery**: Recovering from tool use errors
5. **User preference adaptation**: Adapting tool use based on user preferences

### 6.3 Sample Tool Use Evaluation Script

```python
from qwen_agent.agents import Assistant

# Function to evaluate tool use capabilities
def evaluate_tool_use(task_description, available_tools, expected_tool_sequence):
    """
    Evaluates the model's tool use capabilities on a specific task.
    
    Args:
        task_description: Description of the task to perform
        available_tools: List of tools available to the agent
        expected_tool_sequence: Expected sequence of tools to use
        
    Returns:
        Dictionary with evaluation results
    """
    # Configure LLM
    llm_config = {
        'model': 'Qwen3-235B-A22B',
        'model_server': 'http://localhost:8000/v1',
        'api_key': 'EMPTY',
    }
    
    # Initialize agent with specified tools
    agent = Assistant(llm=llm_config, function_list=available_tools)
    
    # Create the task message
    messages = [{'role': 'user', 'content': task_description}]
    
    # Run the agent and track tool usage
    tool_usage = []
    for response in agent.run(messages=messages):
        # Extract tool usage information if available
        if 'function_call' in response.get('choices', [{}])[0].get('message', {}):
            function_call = response['choices'][0]['message']['function_call']
            tool_usage.append({
                'name': function_call.get('name'),
                'arguments': function_call.get('arguments')
            })
    
    # Calculate tool selection accuracy
    correct_tools = 0
    for i, expected_tool in enumerate(expected_tool_sequence):
        if i < len(tool_usage) and tool_usage[i]['name'] == expected_tool:
            correct_tools += 1
    
    tool_accuracy = correct_tools / len(expected_tool_sequence) if expected_tool_sequence else 0
    
    # Record final response
    final_response = agent.get_final_response(messages)
    
    return {
        'task': task_description,
        'tool_usage': tool_usage,
        'expected_tools': expected_tool_sequence,
        'tool_accuracy': tool_accuracy,
        'tool_count_match': len(tool_usage) == len(expected_tool_sequence),
        'final_response': final_response
    }

# Define evaluation tasks with expected tool usage
evaluation_tasks = [
    {
        'description': 'What is the current time in Tokyo, Japan?',
        'tools': ['time'],
        'expected_sequence': ['time']
    },
    {
        'description': 'Write a Python function to calculate the Fibonacci sequence and test it with n=10.',
        'tools': ['code_interpreter'],
        'expected_sequence': ['code_interpreter']
    },
    {
        'description': 'Get the weather in New York and then convert the temperature from Fahrenheit to Celsius.',
        'tools': ['fetch', 'code_interpreter'],
        'expected_sequence': ['fetch', 'code_interpreter']
    },
    {
        'description': 'Find recent news about artificial intelligence from reputable sources, summarize the key points, and create a table of the main developments.',
        'tools': ['fetch', 'code_interpreter'],
        'expected_sequence': ['fetch', 'code_interpreter', 'code_interpreter']
    }
]

# Evaluate each task with thinking and non-thinking modes
thinking_results = []
non_thinking_results = []

# Configure available tools
available_tools = [
    {'mcpServers': {
        'time': {
            'command': 'uvx',
            'args': ['mcp-server-time']
        },
        'fetch': {
            'command': 'uvx',
            'args': ['mcp-server-fetch']
        }
    }},
    'code_interpreter'
]

# Evaluate with thinking mode
for task in evaluation_tasks:
    # Configure LLM with thinking mode
    llm_config = {
        'model': 'Qwen3-235B-A22B',
        'model_server': 'http://localhost:8000/v1',
        'api_key': 'EMPTY',
        'generate_cfg': {
            'thought_in_content': True,
        },
    }
    
    result = evaluate_tool_use(
        task['description'],
        available_tools,
        task['expected_sequence']
    )
    result['mode'] = 'thinking'
    thinking_results.append(result)

# Evaluate with non-thinking mode
for task in evaluation_tasks:
    # Configure LLM with non-thinking mode
    llm_config = {
        'model': 'Qwen3-235B-A22B',
        'model_server': 'http://localhost:8000/v1',
        'api_key': 'EMPTY',
        'generate_cfg': {
            'thought_in_content': False,
        },
    }
    
    result = evaluate_tool_use(
        task['description'],
        available_tools,
        task['expected_sequence']
    )
    result['mode'] = 'non-thinking'
    non_thinking_results.append(result)

# Analyze results
thinking_accuracy = sum(r['tool_accuracy'] for r in thinking_results) / len(thinking_results)
non_thinking_accuracy = sum(r['tool_accuracy'] for r in non_thinking_results) / len(non_thinking_results)

print(f"Thinking mode tool accuracy: {thinking_accuracy:.2f}")
print(f"Non-thinking mode tool accuracy: {non_thinking_accuracy:.2f}")
```

## 7.
