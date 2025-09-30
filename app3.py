import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()

# Get the Groq API key from environment variables
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Data Analysis and Business Strategy System Prompt (Base for Zero-Shot)
DATA_ANALYSIS_SYSTEM_PROMPT = """
You are a data analysis and business strategy assistant. Perform one of the following tasks based on the user’s request:

Summarize survey results into actionable insights.

Generate a SWOT analysis for a startup.

Write SQL queries from plain English requests.

Create a data-driven marketing campaign outline.
"""

# Five-Shot Prompt (Includes examples)
FIVE_SHOT_SYSTEM_PROMPT = """
You are a data analysis and business strategy assistant. Below are examples of how to complete different requests. Follow the same style and structure when answering new requests.

Example 1 (Survey Summarization):
User: 'Summarize survey results where 70% of employees prefer remote work, 20% hybrid, and 10% onsite.'
Output: 'Majority (70%) prefer remote work, indicating flexibility is highly valued. A smaller portion (20%) prefers hybrid, showing moderate demand for office access. Onsite-only is least preferred (10%), suggesting resistance to traditional setups.'

Example 2 (SWOT Analysis):
User: 'Generate a SWOT analysis for a food delivery startup.'
Output:

Strengths: Fast delivery, wide coverage area

Weaknesses: Limited brand awareness, high operational costs

Opportunities: Growing demand for online food services

Threats: Intense competition, regulatory restrictions

Example 3 (SQL from Plain English):
User: 'Find all customers who spent more than $500 last month.'
Output:

SELECT customer_id, SUM(amount) AS total_spent
FROM transactions
WHERE transaction_date >= '2025-09-01' AND transaction_date <= '2025-09-30'
GROUP BY customer_id
HAVING SUM(amount) > 500;

Example 4 (Marketing Campaign Outline):
User: 'Create a marketing campaign outline for a new fitness app.'
Output:

Target Audience: Young professionals, 20–35 years old

Channels: Instagram, TikTok, Influencer partnerships

Strategy: Free 7-day trial, referral program

KPI: Number of downloads, subscription conversion rate

Example 5 (Survey Summarization):
User: 'Summarize customer satisfaction survey: 60% satisfied, 25% neutral, 15% dissatisfied.'
Output: 'Most customers (60%) are satisfied, showing positive reception. Neutral responses (25%) highlight room for improvement. Dissatisfaction (15%) should be addressed to prevent churn.'
"""

# Test User Request
test_request = "Generate a SWOT analysis for a small e-commerce startup focusing on eco-friendly clothing."

# Initialize Groq LLM
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant",
    temperature=0.7,
    max_tokens=1000
)

# Test Zero-Shot Prompt
print("--- Zero-Shot Prompt Test ---")
try:
    messages = [
        {"role": "system", "content": DATA_ANALYSIS_SYSTEM_PROMPT},
        {"role": "user", "content": test_request}
    ]
    response = llm.invoke(messages)
    print(response.content)
except Exception as e:
    print(f"Error: {str(e)}")

# Test Five-Shot Prompt
print("\n--- Five-Shot Prompt Test ---")
try:
    messages = [
        {"role": "system", "content": FIVE_SHOT_SYSTEM_PROMPT},
        {"role": "user", "content": test_request}
    ]
    response = llm.invoke(messages)
    print(response.content)
except Exception as e:
    print(f"Error: {str(e)}")
