from transformers import AutoTokenizer
import time

# prompt = """You are an expert urban planner and cost estimator with deep knowledge of Paris, France. I need you to provide a comprehensive analysis of what it would cost to hire professional window cleaners to clean all the windows in Paris.

# Consider the following factors in your detailed estimate:
# 1. The total number of buildings and windows in Paris (both residential and commercial)
# 2. Different types of buildings (apartments, offices, shops, historical buildings, etc.)
# 3. The varying heights and accessibility of buildings
# 4. Labor costs for professional window cleaners in Paris
# 5. Equipment and safety requirements for high-rise buildings
# 6. Seasonal variations and weather considerations
# 7. Time estimates for completion
# 8. Any special considerations for historical or landmark buildings

# Please provide your estimate in US Dollars, breaking down the major cost components. Also include any assumptions you're making and potential challenges that could affect the final cost."""

# Load lorem ipsum text from file
with open("./input/lorem.txt", "r", encoding="utf-8") as f:
    prompt = f.read()

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-4-Scout-17B-16E-Instruct", use_fast=True)

# Manual formatting with LLaMA's actual special tokens
user_message = "Please list the top 3 programming languages in 2024."
assistant_message = "Here are the top 3 programming languages in 2024:\n\n1. **Python**: Widely used for AI/ML\n2. **JavaScript**: Essential for web development\n3. **TypeScript**: Like JS, but with types."

# Use the ACTUAL special tokens (not the chat template format)
formatted_prompt = f"<|begin_of_text|>{user_message}<|eot|>{assistant_message}<|eot|><|end_of_text|>"
incorrectly_formatted_prompt = f"<<begin_of_text|>{user_message}<<eot|>{assistant_message}<<eot|><<end_of_text|>"
prompt_long = "<|begin_of_text|>What are the main differences between Python and JavaScript?<|eot|>Here are the key differences between Python and JavaScript:\n\n**1. Syntax and Readability**\n- Python: Uses indentation for code blocks, very readable\n- JavaScript: Uses curly braces, more C-like syntax\n\n**2. Type System**\n- Python: Dynamically typed with optional type hints\n- JavaScript: Dynamically typed, TypeScript adds static typing\n\n**3. Primary Use Cases**\n- Python: Data science, AI/ML, backend development, automation\n- JavaScript: Web development (frontend/backend), mobile apps\n\n**4. Performance**\n- Python: Generally slower, interpreted language\n- JavaScript: V8 engine makes it quite fast for web applications<|eot|>Can you give me a concrete example of the syntax differences?<|eot|>Absolutely! Here are concrete examples showing the syntax differences:\n\n**Python Example:**\n```python\ndef calculate_factorial(n):\n    if n <= 1:\n        return 1\n    else:\n        return n * calculate_factorial(n - 1)\n\n# Usage\nresult = calculate_factorial(5)\nprint(f\"Factorial of 5 is: {result}\")\n\n# List comprehension\nnumbers = [1, 2, 3, 4, 5]\nsquares = [x**2 for x in numbers if x % 2 == 0]\nprint(squares)  # Output: [4, 16]\n```\n\n**JavaScript Example:**\n```javascript\nfunction calculateFactorial(n) {\n    if (n <= 1) {\n        return 1;\n    } else {\n        return n * calculateFactorial(n - 1);\n    }\n}\n\n// Usage\nconst result = calculateFactorial(5);\nconsole.log(`Factorial of 5 is: ${result}`);\n\n// Array methods\nconst numbers = [1, 2, 3, 4, 5];\nconst squares = numbers.filter(x => x % 2 === 0).map(x => x**2);\nconsole.log(squares); // Output: [4, 16]\n```\n\n**Key Differences Highlighted:**\n- Python uses indentation vs JavaScript's curly braces `{}`\n- Python's `def` vs JavaScript's `function`\n- Python's f-strings vs JavaScript's template literals\n- Python's list comprehensions vs JavaScript's array methods<|eot|>Which one would you recommend for a beginner programmer?<|eot|>For a beginner programmer, I'd generally recommend **Python** for these reasons:\n\n**Why Python is Great for Beginners:**\n\n1. **Readable Syntax**: Python's syntax closely resembles English, making it intuitive\n   ```python\n   if age >= 18:\n       print(\"You can vote!\")\n   ```\n\n2. **Less Boilerplate**: You can write functional programs with minimal setup\n   ```python\n   name = input(\"What's your name? \")\n   print(f\"Hello, {name}!\")\n   ```\n\n3. **Excellent Learning Resources**: Tons of tutorials, books, and courses designed for beginners\n\n4. **Versatile Applications**: Once you learn Python, you can:\n   - Build web applications (Django, Flask)\n   - Analyze data (pandas, NumPy)\n   - Create AI/ML models (TensorFlow, PyTorch)\n   - Automate tasks (scripting)\n   - Develop games (Pygame)\n\n5. **Gentle Learning Curve**: Focus on problem-solving rather than complex syntax\n\n**However, JavaScript might be better if you:**\n- Want to see immediate visual results (web pages)\n- Are specifically interested in web development\n- Prefer learning through interactive projects\n\n**My Recommendation**: Start with Python to learn programming fundamentals, then add JavaScript when you want to build web applications. This gives you a solid foundation plus practical web skills!\n\nWhat type of projects are you most interested in creating?<|eot|>I'm interested in building web applications. Should I still start with Python?<|eot|>Given your interest in web applications, this changes my recommendation! Here's what I'd suggest:\n\n**For Web Development, Consider Starting with JavaScript:**\n\n**Advantages of JavaScript-First for Web Development:**\n1. **Immediate Visual Feedback**: See your changes instantly in the browser\n2. **One Language, Full Stack**: JavaScript works for both frontend and backend (Node.js)\n3. **No Setup Required**: Just open a browser and start coding\n4. **Huge Ecosystem**: React, Vue, Angular for frontend; Express, Next.js for backend\n5. **High Demand**: Web developers are in high demand\n\n**Learning Path for Web Development:**\n```\n1. HTML + CSS (structure and styling)\n2. JavaScript fundamentals\n3. DOM manipulation\n4. Frontend framework (React recommended)\n5. Backend with Node.js/Express\n6. Database integration (MongoDB/PostgreSQL)\n```\n\n**Sample First Project** (you can build this in days):\n```html\n<!DOCTYPE html>\n<html>\n<head><title>Todo App</title></head>\n<body>\n    <h1>My Todo List</h1>\n    <input id=\"todoInput\" placeholder=\"Add a task...\">\n    <button onclick=\"addTodo()\">Add</button>\n    <ul id=\"todoList\"></ul>\n    \n    <script>\n    function addTodo() {\n        const input = document.getElementById('todoInput');\n        const list = document.getElementById('todoList');\n        const li = document.createElement('li');\n        li.textContent = input.value;\n        list.appendChild(li);\n        input.value = '';\n    }\n    </script>\n</body>\n</html>\n```\n\n**Alternative: Python for Web (Still Valid!)**\n- Django/Flask are excellent for web backends\n- Python + JavaScript frontend is a common combination\n- Many successful web companies use Python (Instagram, Spotify, Dropbox)\n\n**My Updated Recommendation**: Start with JavaScript since you want immediate web results, but don't completely ignore Python—you might use it later for backend services, data processing, or AI features in your web apps!\n\nWould you like me to suggest some specific first projects to try?<|eot|><|end_of_text|>"

t0 = time.time()
tokens = tokenizer.encode(prompt_long)
t1 = time.time()
print(f"Time taken: {(t1 - t0) * 1000:.2f}ms")
print(f"Token count: {len(tokens)}")
# for token in tokens:
    # print(token)

# # Check first few tokens
# print("\nFirst 10 tokens decoded:")
# for i, token_id in enumerate(tokens[:10]):
#     decoded = tokenizer.decode([token_id])
#     print(f"Token {token_id}: '{decoded}'")

# # Check special tokens
# print("Special tokens in tokenizer:")
# print(f"BOS token: '{tokenizer.bos_token}' -> ID: {tokenizer.bos_token_id}")
# print(f"EOS token: '{tokenizer.eos_token}' -> ID: {tokenizer.eos_token_id}")
# print(f"PAD token: '{tokenizer.pad_token}' -> ID: {tokenizer.pad_token_id}")

# # Check if [INST] and [/INST] are special tokens
# special_tokens = tokenizer.special_tokens_map
# print(f"All special tokens: {special_tokens}")

# # Tokenize individual special tokens
# print("\nIndividual token IDs:")
# print(f"<s> -> {tokenizer.encode('<s>', add_special_tokens=False)}")
# print(f"[INST] -> {tokenizer.encode('[INST]', add_special_tokens=False)}")
# print(f"[/INST] -> {tokenizer.encode('[/INST]', add_special_tokens=False)}")
# print(f"</s> -> {tokenizer.encode('</s>', add_special_tokens=False)}")

# # Decode first few tokens to see what they are
# print("\nFirst 10 tokens decoded:")
# for i, token_id in enumerate(tokens[:10]):
#     decoded = tokenizer.decode([token_id])
#     print(f"Token {token_id}: '{decoded}'")

# print("\nLast 5 tokens decoded:")
# for i, token_id in enumerate(tokens[-5:]):
#     decoded = tokenizer.decode([token_id])
#     print(f"Token {token_id}: '{decoded}'")
