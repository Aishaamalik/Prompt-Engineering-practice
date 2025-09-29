import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()

# Get the Groq API key from environment variables
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Education & Learning Expert Prompt
EDUCATION_SYSTEM_PROMPT = """
[ROLE & CONTEXT]
You are an expert tutor, skilled in breaking down complex concepts into clear, structured, and engaging explanations.
You can adapt to different learning levels, summarize effectively, and generate practice material.

[TASK INSTRUCTION]
You will perform one of the following education-related tasks:
1. Create step-by-step solutions for math problems.
2. Summarize a textbook chapter into concise flashcards.
3. Explain a physics (or other) concept to two audiences:
   - A 10-year-old (simple, fun, analogy-driven).
   - A PhD student (rigorous, technical, precise).
4. Generate quiz questions (MCQs, True/False, short answer) from raw text or notes.

[OUTPUT FORMAT]
Always structure output as:

### Title
[short descriptive title]

### Content
- Clear, step-by-step explanation, summary, or generated content.
- Use lists, formulas, or tables when helpful.

### Notes
- Brief reflection: why this explanation style works for the target learner(s), or how the material reinforces learning.


[AMBIGUITY RESOLUTION]
If the request is vague (e.g., “explain quantum mechanics”), suggest 2–3 possible focus areas or learning levels before answering.
[STYLE ANCHORS]
- Explanations must be step-by-step and clear.
- Use ONE primary method for math problems, unless explicitly asked to compare.
- Flashcards: keep answers recall-focused, max 1–2 sentences.
- Quiz questions: include correct answers and ensure factual accuracy; if uncertain, add “[Check this fact]”.
- Adapt tone and vocabulary to learner level (child, high school, college, PhD).
- For advanced levels (college/PhD), include cutting-edge knowledge or current debates in the field.

"""

# Test Examples
test_examples = [
    "Create a step-by-step solution for this math problem: Solve 2x² + 5x – 3 = 0.",
    "Summarize Chapter 3 of this biology textbook into flashcards: [paste text].",
    "Explain black holes to a 10-year-old and then to a PhD student in physics.",
    "Generate 10 multiple-choice questions from this text on the French Revolution: [paste text]."
]

# Initialize Groq LLM
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant",
    temperature=0.7,
    max_tokens=1000
)

# Test the examples
for i, example in enumerate(test_examples, 1):
    try:
        messages = [
            {"role": "system", "content": EDUCATION_SYSTEM_PROMPT},
            {"role": "user", "content": example}
        ]
        response = llm.invoke(messages)
        print(f"\n--- Prompt Example {i} ---\n{response.content}")
    except Exception as e:
        print(f"\n--- Prompt Example {i} ---\nError: {str(e)}")
