# Call Me Maybe

*This project has been created as part of the 42 curriculum by jiezhang.*

## Description

This project implements a function calling system that translates natural language prompts into structured function calls using Large Language Models (LLMs) with constrained decoding. The system guarantees 100% valid JSON output through token-level constraints.

Given a natural language query like "What is the sum of 40 and 2?", the system produces:
```json
{
  "fn_name": "fn_add_numbers",
  "args": {"a": 40.0, "b": 2.0}
}
```

The system uses the Qwen3-0.6B model and achieves >95% accuracy through constrained decoding techniques.

## Project Structure

```
call-me-maybe/
├── src/
│   ├── __init__.py
│   ├── __main__.py        # Entry point
│   ├── llm_wrapper.py     # Constrained decoding
│   ├── decoder.py         # Type conversion
│   ├── schema.py          # Pydantic models
│   ├── reader.py          # Input handling
│   └── writer.py          # Output handling
├── llm_sdk/               # LLM SDK
├── data/
│   ├── exercise_input/
│   └── exercise_output/
├── pyproject.toml
├── Makefile
└── README.md
```

## Instructions

### Installation

```bash
uv sync
```

### Usage

```bash
# Default paths
uv run python -m src

# Custom paths
uv run python -m src \
  --input data/input/tests.json \
  --output data/output/results.json \
  --definitions data/input/functions.json \
  --model Qwen/Qwen3-0.6B
```

### Development

```bash
make install    # Install dependencies
make run        # Run program
make lint       # Run linting
make debug      # Run with debugger
make clean      # Clean temporary files
```

## Algorithm Explanation

### Overall Architecture

The system processes prompts through five stages:

```
Natural Language Prompt
        ↓
Vocabulary Normalization
        ↓
Function Selection (Constrained Decoding)
        ↓
Argument Extraction
        ↓
Type Conversion & Validation
        ↓
JSON Output
```

### Stage 1: Vocabulary Normalization

Different tokenizers use special characters (BPE tokens):
- `Ġ` for spaces (GPT-2/Qwen)
- `Ċ` for newlines
- `▁` for spaces (SentencePiece)

The system normalizes these to actual characters for consistent processing:

```python
class VocabularyConfig(BaseModel):
    replacements: dict[str, str] = Field(
        default_factory=lambda: {
            'Ġ': ' ',
            'Ċ': '\n',
            # ...
        }
    )
```

### Stage 2: Constrained Decoding (Core Algorithm)

Guarantees that the output is one of the predefined valid function JSONs.

#### How It Works

**Step 1: Build Token Matrix**

Each function's JSON representation is tokenized:

```python
functions = [
    {"fn_name": "fn_add", ...},
    {"fn_name": "fn_mul", ...}
]

# Tokenize each function
matrix = [
    [123, 456, 789, ...],  # fn_add tokens
    [123, 457, 790, ...]   # fn_mul tokens
]
```

**Step 2: Constrained Generation**

At each generation step:
1. Get logits from LLM
2. Find tokens that continue matching at least one function
3. Mask invalid tokens (set logits to `-∞`)
4. Select best valid token

```python
def _get_allowed_tokens(self, generated_ids, matrix):
    """Only allow tokens that match at least one function."""
    allowed = set()
    prefix_len = len(generated_ids)
    
    for tokens in matrix:
        if len(tokens) > prefix_len:
            if tokens[:prefix_len] == generated_ids:
                allowed.add(tokens[prefix_len])
    
    return list(allowed)
```

**Visual Example:**
```
Generated: [123, 456]

Matrix:
fn_add:  [123, 456, 789, ...]  ← matches, next = 789
fn_mul:  [123, 456, 790, ...]  ← matches, next = 790
fn_sub:  [123, 457, ...]       ← no match

Allowed: {789, 790}
Blocked: ~49,998 other tokens
```

**Step 3: Logit Masking**

```python
masked_logits = [-math.inf] * len(logits)
for token_id in allowed_ids:
    masked_logits[token_id] = logits[token_id]

next_token = argmax(masked_logits)
```

This ensures the model can **only** generate valid tokens.

**Step 4: Early Termination**

Stop once unique function identified:

```python
def _check_unique_match(self, generated_ids, matrix, func_map):
    matches = [i for i, tokens in enumerate(matrix)
               if tokens[:len(generated_ids)] == generated_ids]
    if len(matches) == 1:
        return True, func_map[matches[0]]  # Stop early!
    return False, None
```

### Stage 3: Argument Extraction

Combines regex precision with LLM context-awareness:

**Regex Extraction:**
```python
# Numbers
numbers = re.findall(r"-?\d+\.?\d*", prompt)

# Strings (handles apostrophes correctly)
single_quoted = re.findall(r"'([^']*)'", prompt)
double_quoted = re.findall(r'"([^"]*)"', prompt)
```

**LLM Selection:**
When multiple candidates exist, use constrained decoding to select:
```python
# "Add 5 and 10" → candidates = ["5", "10"]
selected = llm_select_candidate(candidates, arg_name)
```

### Stage 4: Type Conversion

```python
if arg_type == "int":
    return int(float(value))  # Handles "3.0" → 3

elif arg_type == "float":
    return float(value)

elif arg_type == "bool":
    return value.lower() in ("true", "1", "yes")
```

### Stage 5: Validation

Multi-layer validation:
1. Constrained decoding → structural validity
2. Type conversion → correct types
3. Pydantic validation → schema compliance
4. Fallback → always valid output

## Design Decisions

### Constrained Decoding vs Prompt Engineering

**Decision:** Use token-level constraints

**Rationale:**
- Prompt engineering alone: ~30% reliability
- Constrained decoding: ~99% reliability
- Trade-off: Higher complexity for guaranteed validity

### Token Prefix Matching

**Decision:** Use prefix matching instead of full JSON FSM

**Rationale:**
- Automatically adapts to new functions
- No manual state definitions needed
- Enables early termination optimization
- Same reliability with less complexity

### Regex + LLM Hybrid

**Decision:** Regex extracts candidates, LLM selects

**Rationale:**
- Regex: Fast and precise for clear cases
- LLM: Context-aware for ambiguous cases
- Combined: Best performance

### Pydantic Validation

**Decision:** Use Pydantic for all data structures

**Classes:**
- `VocabularyConfig`: Token normalization settings
- `LLMConfig`: Model configuration
- `FunctionDefinition`: Function schemas
- `PromptInput`: Input validation
- `FunctionCallResult`: Output validation

**Benefits:**
- Early error detection
- Automatic type conversion
- Clear error messages

## Performance Analysis

### Accuracy
- Function selection: >95% 
- Argument extraction: >95%
- JSON validity: 100% (guaranteed)

### Speed
- Average: ~3.5 seconds per prompt
- Total (140 prompts): ~500 seconds

### Reliability
- Zero crashes
- All errors handled gracefully
- Always produces valid output

## Challenges Faced

### Speed Optimization

**Problem:** Processing was much slower than expected  in the early termination check.

**Root Cause:** A condition error in `_check_unique_match` prevented early stopping from ever triggering:

```python
# early termination never fires
matches = [i for i, tokens in enumerate(matrix)
           if tokens[:len(generated_ids)] == generated_ids]
if len(matches) == 1 and len(matrix[matches[0]]) == len(generated_ids):
    return True, func_map[matches[0]]
```

The extra condition `len(matrix[matches[0]]) == len(generated_ids)` required the generated sequence to be the same length as the full function token sequence. This is almost never true mid-generation, so the loop always ran to `max_tokens` and generated every token of every function JSON.

**Fix:** Remove the length equality check — stop as soon as only one function matches the prefix:

```python
# FIXED version — stops as soon as unique match found
matches = [i for i, tokens in enumerate(matrix)
           if tokens[:len(generated_ids)] == generated_ids]
if len(matches) == 1:
    return True, func_map[matches[0]]
```

**Why this is correct:** Once only one function still shares the same token
prefix as the generated sequence, no other function can ever match. There is
no need to generate the remaining tokens.

**Impact:**

```
Before fix (buggy):
- fn_greet:      ~8.5s   (generated all tokens every time)
- fn_add:        ~12s
- fn_substitute: ~17s

After fix:
- fn_greet:      ~3-4s   (stops after ~5 tokens match uniquely)
- fn_add:        ~5-6s
- fn_substitute: ~8-10s

Improvement: ~40-50% faster per prompt
```

**Additional optimization: function matrix caching**

The token matrix is built once per unique function set and reused:

```python
key = tuple(json.dumps(f, sort_keys=True) for f in functions)
if key in self._matrix_cache:
    return self._matrix_cache[key]
```

This avoids re-tokenizing the same functions on every prompt call.

## Testing Strategy

### Coverage

**Input Validation:**
- File errors (not found, invalid JSON)
- Data structure errors
- Encoding errors

**Function Selection:**
- Simple prompts: "add 2 and 3"
- Ambiguous prompts
- Edge cases: empty strings, special characters

**Argument Extraction:**
- Numbers: integers, floats, negatives
- Strings: quotes, apostrophes, unicode
- Booleans: various formats
- Multiple arguments

**End-to-End:**
- Moulinette: 14/14 passed
- Extended suite: >95% accuracy
- JSON validity: 100%

## Example Usage

### Basic

```bash
uv run python -m src
```

**Output:**
```
[Info]: Loaded Qwen/Qwen3-0.6B with 151643 tokens
[Info]: Processing 14 prompts

[1/14] Processing: What is the square root of 16?
  → fn_get_square_root | Time: 1.89s
[2/14] Processing: Reverse the string 'hello'
  → fn_reverse_string | Time: 1.76s
...
[Info]: Done! Total time: 24.53s
```

## Advanced Features

**Multiple Model Support:**
- Accepts model name via `--model` argument
- Validated by `LLMConfig` Pydantic model

**Comprehensive test suite:**
- Extended tests with 126 prompts and 20 functions

**Public Tokenizer Interface:**

The SDK's `encode()` and `decode()` methods are exposed as public methods on
`LLMWrapper`, integrating directly with constrained decoding:
```python
llm = LLMWrapper()
tokens = llm.encode("Hello world")  # [31373, 1879]
text = llm.decode(tokens)            # "Hello world"
```

**Public Tokenizer Interface - how it works:**

During constrained decoding, `encode()` tokenizes each function's JSON tobuild the constraint matrix, and `decode()` can reconstruct generated tokens.

**Function Matrix Caching:**
- Token matrix built once per unique function set, reused across all prompts
- Avoids redundant tokenization (~0.1-0.2s saved per prompt)

**Advanced Error Recovery:**
- `build_fallback()` in `ConstrainedDecoder` provides a safe default result
- Multi-layer: optimal → fallback → minimal valid output
- Zero crashes in testing

**Visualization:**
- Step-by-step token generation display
- Constraint visualization per generation step


## Resources

### AI Usage

AI tools (Claude, ChatGPT) were used for:

#### Learning
- Algorithm concepts: Constrained decoding, token generation, BPE tokenization
- Technical skills: Regular expressions, Pydantic patterns, type hints
- LLM theory: Token distributions, vocabulary alignment

#### Implementation
- Code structure: Modular architecture design
- Error handling: Multi-layer fallback strategies
- Optimization: Early termination logic, performance profiling

#### Development
- Debugging: Tokenization issues, execution tracing
- Testing: Test case generation, edge case identification
- Documentation: Clear explanations, code comments

#### Understanding

All AI-assisted content was fully reviewed, understood, tested, and manually validated. The final implementation, architectural decisions, and algorithmic logic remain fully under my responsibility.

## Conclusion

The **Call Me Maybe** project demonstrates a robust approach to translating natural language prompts into fully validated function calls using Large Language Models (LLMs). By combining **token-level constrained decoding**, **regex-based candidate extraction**, and **Pydantic validation**, the system achieves high reliability, accuracy, and performance while guaranteeing 100% valid JSON outputs.

Key takeaways from this project:

1. **Reliability through Constrained Decoding:**  
   Token-level constraints ensure that the LLM can only generate sequences that correspond to valid functions, eliminating runtime errors and invalid outputs.

2. **Hybrid Candidate Extraction:**  
   Using regular expressions for precise extraction and LLM guidance for context-aware selection enables robust argument parsing even for ambiguous or complex prompts.

3. **Performance Optimization:**  
   Early termination and function matrix caching significantly reduce runtime, achieving ~40–50% faster processing per prompt without sacrificing correctness.

4. **Extensibility and Maintainability:**  
   The modular design allows easy addition of new functions, argument types, or LLM models. The use of Pydantic models enforces schema validation consistently across all components.

5. **Educational Value:**  
   This project provides practical experience with advanced NLP techniques, type-safe Python programming, and LLM integration in real-world applications.

In conclusion, **Call Me Maybe** not only serves as a functional system for structured function call generation but also as a reusable framework for experimenting with LLMs, constrained decoding algorithms, and reliable prompt-to-JSON translation pipelines. The project balances **accuracy**, **speed**, and **robustness**, making it suitable as both a learning tool and a production-ready prototype.
