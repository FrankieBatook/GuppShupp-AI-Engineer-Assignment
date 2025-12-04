import streamlit as st
import json
from src.llm_factory import LLMFactory
from src.memory_agent import MemoryAgent
from data.conversation_agent import ConversationAgent

# --- Page Config ---
st.set_page_config(page_title="GuppShupp AI Engine",
                   layout="wide", page_icon="🧠")

# --- Sidebar: Model Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")

    provider = st.selectbox(
        "Select AI Provider",
        ["Google", "Anthropic", "OpenRouter"],
        index=0
    )

    api_key = st.text_input(f"{provider} API Key", type="password")

    # Dynamic model selection based on provider
    model_map = {
        "Google": ["gemini-2.5-pro", "gemini-2.5-flash"],
        "Anthropic": ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229"],
        "OpenRouter": ["deepseek/deepseek-r1", "anthropic/claude-3.5-sonnet", "meta-llama/llama-3.3-70b-instruct"]
    }

    model_name = st.selectbox(
        "Select Model", model_map.get(provider, ["default"]))

    st.divider()
    st.info("Note: For OpenRouter, ensure your key has credits.")

# --- Main UI ---
st.title("🧠 GuppShupp: Agentic Personality Engine")

if not api_key:
    st.warning(
        f"⚠️ Please enter your {provider} API key in the sidebar to initialize the agents.")
    st.stop()

# --- Initialize Agents ---
try:
    # We create a single LLM instance to be shared (or you could create two different ones)
    main_llm = LLMFactory.create_llm(provider, api_key, model_name)

    memory_agent = MemoryAgent(main_llm)
    conversation_agent = ConversationAgent(main_llm)

except Exception as e:
    st.error(f"Failed to initialize LLM: {e}")
    st.stop()

# --- Session State Management ---
if "memory_profile" not in st.session_state:
    st.session_state.memory_profile = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Layout ---
tab1, tab2 = st.tabs(["📊 Memory Extraction", "💬 Personality Chat"])

# TAB 1: MEMORY EXTRACTION
with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Raw User Logs")
        # Load default data
        with open("data/data/sample_chat.json", "r") as f:
            default_data = json.load(f)

        input_text = st.text_area(
            "Paste Chat History (JSON or Text)",
            value=json.dumps(default_data, indent=2),
            height=400
        )

        if st.button("🔍 Analyze & Extract Memory", type="primary"):
            with st.spinner("Agent is reading chat logs..."):
                # Parse input to list
                try:
                    raw_msgs = json.loads(input_text) if input_text.startswith(
                        "[") else input_text.split("\n")
                    profile = memory_agent.extract_from_history(raw_msgs)
                    st.session_state.memory_profile = profile
                    st.success("Extraction Complete!")
                except Exception as e:
                    st.error(f"Error processing input: {e}")

    with col2:
        st.subheader("Structured Memory Profile")
        if st.session_state.memory_profile:
            st.json(st.session_state.memory_profile)
        else:
            st.info("Run the extraction to see the Agent's internal state.")

# TAB 2: CONVERSATIONAL AGENT
with tab2:
    st.subheader("Test the Personality Engine")

    if not st.session_state.memory_profile:
        st.warning(
            "⚠️ Please run the 'Memory Extraction' first so the agent knows who you are.")
    else:
        # Persona Selector
        selected_persona = st.radio(
            "Select Active Persona:",
            ["Calm Mentor", "Witty Friend", "Therapist"],
            horizontal=True
        )

        # Chat Interface
        user_input = st.chat_input("Say something to your AI friend...")

        # Display History
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if user_input:
            # User Message
            st.session_state.chat_history.append(
                {"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # AI Response
            with st.spinner(f"{selected_persona} is typing..."):
                response = conversation_agent.generate_response(
                    user_input,
                    st.session_state.memory_profile,
                    selected_persona
                )

            st.session_state.chat_history.append(
                {"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
