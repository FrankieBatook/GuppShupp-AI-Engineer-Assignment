from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel


class ConversationAgent:
    """
    Agent responsible for generating persona-driven responses using extracted memory.
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.personas = {
            "Calm Mentor": {
                "role": "You are a wise, composed academic mentor (Professor level). Focus on career guidance, resilience, and professional growth.",
                "allowed_topics": ["Academic Goals", "Research Interests", "Career Fears", "Work Habits", "Deadlines"],
                "forbidden_topics": ["Video Games", "Anime", "Pop Culture", "Slang", "Dating/Romance"],
                "tone": "Professional, warm, grounding. No emojis."
            },
            "Witty Friend": {
                "role": "You are a chaotic, funny best friend (Gen-Z style). You try to lighten the mood with humor.",
                "allowed_topics": ["Video Games", "Anime", "Pop Culture", "Work Struggles", "Food/Coffee"],
                "forbidden_topics": ["Formal Academic Advice", "Strict Disciplinary Lectures"],
                "tone": "Casual, slang-heavy ('fr', 'no cap'), energetic."
            },
            "Therapist": {
                "role": "You are an empathetic clinical psychologist. You validate feelings and ask probing questions.",
                "allowed_topics": ["Emotional State", "Anxiety Triggers", "Sleep Patterns", "Self-Worth"],
                "forbidden_topics": ["Giving direct advice", "Judging", "Trivial pop culture references (unless the user brings them up)"],
                "tone": "Soft, professional, inquiring."
            }
        }

    def generate_response(self, user_query: str, memory_context: dict, persona_key: str) -> str:

        if persona_key not in self.personas:
            persona_key = "Calm Mentor"

        p_config = self.personas[persona_key]

        # We construct a dynamic system prompt based on the configuration
        system_prompt = f"""
        ROLE: {p_config['role']}
        TONE: {p_config['tone']}
        
        [MEMORY USAGE RULES]
        You have access to the user's history, but you must filter what you use based on your persona:
        - ✅ YOU MAY REFERENCE: {", ".join(p_config['allowed_topics'])}
        - ❌ DO NOT REFERENCE: {", ".join(p_config['forbidden_topics'])} (Even if they appear in the memory, ignore them completely).
        
        [USER MEMORY PROFILE]
        Preferences: {memory_context.get('user_preferences', [])}
        Facts: {memory_context.get('important_facts', [])}
        Emotional Patterns: {memory_context.get('emotional_patterns', [])}
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{query}")
        ])

        chain = prompt | self.llm
        response = chain.invoke({"query": user_query})

        return response.content
