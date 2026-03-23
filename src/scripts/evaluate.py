# scripts/evaluate.py - Evaluate the fine-tuned TSQL text-to-SQL model
#
# Prompts are written as plain-English business questions (non-technical user framing),
# matching the training data format.

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("../outputs/qwen-tsql-specialized", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B", trust_remote_code=True)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

# Prompts written as non-technical business user questions —
# the same framing used in the training data
prompts = [
    "Show me all customers who haven't placed an order in the last 90 days.",
    "What were our top 10 products by total revenue last quarter?",
    "Give me a list of employees in each department along with their average salary.",
    "Which orders from last month are still unpaid?",
    "Show me the running total of sales by month for this year.",
    "Find any duplicate email addresses in our customer list.",
    "What's the average time between when an order is placed and when it ships, by product category?",
    "List all customers whose total spending this year exceeds $10,000, sorted highest to lowest.",
]

for prompt in prompts:
    print("Prompt:", prompt)
    out = generator(prompt, max_new_tokens=300, do_sample=True, temperature=0.3)
    print("Output:\n", out[0]["generated_text"])
    print("\n" + "-" * 60 + "\n")
