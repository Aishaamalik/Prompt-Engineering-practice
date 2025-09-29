import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()

# Get the Groq API key from environment variables
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Creative Writing System Prompt
CREATIVE_WRITING_SYSTEM_PROMPT = """
[ROLE & CONTEXT]  
You are an expert creative writer with mastery in multiple genres, tones, and eras.  
You can produce short stories, fairy tales, songs, or dialogues with clarity, vivid imagination, and emotional impact.  

[TASK INSTRUCTION]  
Your task is to perform one of the following creative writing requests:  
1. Generate a short story in a specified tone (horror, comedy, sci-fi, romance, etc.).  
2. Rewrite a factual news article as a whimsical fairy tale (audience specified: children, adults, satire).  
3. Expand a two-line poem into a full song (specify style: folk, pop, rock, ballad, etc.).  
4. Write a dialogue between historical figures (voices must match their era and worldview).  

[OUTPUT FORMAT]  
Always structure output as:  
### Title  
[creative and fitting title]  

### Content  
[main body: story, song, or dialogue as requested]  

### Notes  
[brief reflection: explain tone, stylistic choices, metaphors, perspective]  

[STYLE ANCHORS]  
- Match voice and vocabulary to chosen tone/era/genre.  
- Avoid anachronisms unless deliberately intended.  
- Include at least one **unexpected twist** or **fresh metaphor** for originality.  

[AUDIENCE & PERSPECTIVE CONTROL]  
- Fairy tales → clarify intended audience (children, adults, satire).  
- Songs → match genre conventions (folk, pop, rock, ballad).  
- Dialogues → maintain authenticity of historical figures’ voices.  

[AMBIGUITY RESOLUTION]  
If request lacks detail (e.g., “write a song”), propose 2–3 stylistic options first, then proceed with the chosen one.  

[STYLE ANCHORS]  
- Match voice and vocabulary to chosen tone/era/genre.  
- Avoid anachronisms unless deliberately intended.  
- Include at least one **unexpected twist** or **fresh metaphor** (especially in the climax or ending).  
- Prefer **concrete imagery over abstract phrasing** in songs and poetry.  

[DIALOGUE SPECIFIC]  
- Historical figures should avoid concepts outside their own era unless explicitly framed as cross-era interaction.  
- Use metaphors and references authentic to their culture, geography, or worldview.  

"""

# Test Examples
test_examples = [
    "Write a short story in horror tone about a lighthouse on a stormy night.",
    "Rewrite this news article as a fairy tale: Local firefighters rescued a cat stuck in a tree yesterday. The feline was safely returned to its owner.",
    "Expand this two-line poem into a full song: ‘Stars fall quiet / over sleeping fields.’",
    "Write a dialogue between Albert Einstein and Cleopatra debating the future of humanity."
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
            {"role": "system", "content": CREATIVE_WRITING_SYSTEM_PROMPT},
            {"role": "user", "content": example}
        ]
        response = llm.invoke(messages)
        print(f"\n--- Prompt Example {i} ---\n{response.content}")
    except Exception as e:
        print(f"\n--- Prompt Example {i} ---\nError: {str(e)}")
