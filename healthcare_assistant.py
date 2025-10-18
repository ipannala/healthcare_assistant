import streamlit as st
from autogen_agentchat.agents import UserProxyAgent, AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import SelectorGroupChat
from autogen_core.tools import FunctionTool
from autogen_agentchat.conditions import TextMentionTermination
from typing import Optional, List, Dict, Any
from functions import retrieve_patient_information,lookup_medical_information,add_medical_records,book_appointment, router_intent
from functools import partial
from openai import RateLimitError
import asyncio
import random


# Set up the OpenAI model client
model = OpenAIChatCompletionClient(
    model='gpt-4o-mini',
    temperature=0
)

# Agent tools
retrieve_patient_information_tool = FunctionTool(retrieve_patient_information, name="retrieve_patient_information", description="Retrieves patient information from patient table")
lookup_medical_information_tool = FunctionTool(lookup_medical_information, name="lookup_medical_information", description="Searches medical information from pubmed journal")
add_medical_records_tool = FunctionTool(add_medical_records, name="add_medical_records", description="Update medical records in the records table")
book_appointment_tool = FunctionTool(book_appointment, name="book_appointment", description="Book an appointment with doctor")

# Agent definitions
retrieve_patient_information_agent = AssistantAgent(
    name='retrieve_patient_information_agent',
    model_client=model,
    tools=[retrieve_patient_information_tool],
    system_message=(
        "You are an expert medical assistant.\n"
        "First call the retrieve_patient_information_tool using the patient ID in this format: retrieve_patient_information(patient ID) function to retrieve patient information.\n"
        "Summarize the medical condition, medications patient is using, and the doctor notes.\n"
        "When you are done, finish your last line with the exact word: TERMINATE")
)

lookup_medical_information_agent = AssistantAgent(
    name="lookup_medical_information_agent",
    model_client=model,
    tools=[lookup_medical_information_tool],
    system_message=(
        "You are an expert medical assistant.\n"
        "First call the lookup_medical_information_tool using the user query in this format: lookup_medical_information(user query) function to search medical journals about the medical question.\n"
        "Summarize the medical journal abstract text in about 6 - 8 sentences. Ensure the important information from the medical journal abstract text is included in the summary.\n"
        "When you are done, finish your last line with the exact word: TERMINATE")
)

add_medical_records_agent = AssistantAgent(
    name="add_medical_records_agent",
    model_client=model,
    tools=[add_medical_records_tool],
    system_message=(
        "You are an expert medical assistant for managing medical records.\n"
        "Retrieve the condition name, medications from the doctor notes.\n"
        "Summarize the doctor notes.\n"
        "Then call the add_medical_records_tool and insert this information in the tool by calling the add_medical_records python function in this format: add_medical_records(patient id, condition names, medications, doctor notes)\n"
        "When you are done, finish your last line with the exact word: TERMINATE"
    )
)

book_appointment_agent = AssistantAgent(
    name="book_appointment_agent",
    model_client=model,
    tools=[book_appointment_tool],
    system_message=(
        "You are an expert medical assistant for booking appointments.\n"
        "Retrieve the patient id, full name of the doctor, and appointment time. Call the book_appointment_tool by calling the book_appointment python function in this format: book_appointment(patient_id, doctor_name, appointment_time)\n"
        "When you are done, finish your last line with the exact word: TERMINATE"
    )
)

group_chat_manager = AssistantAgent(
    name="GroupChatManager",
    model_client=model,
    system_message="You are the group chat manager, responsible for making sure the group conversation is terminated when the user's request has been fulfilled or an exit condition is met.",
    description="This agent is a group chat manager.",
)

def _extract_text_from_thread(thread) -> str:
    """Return the most recent message text from a thread-like payload."""
    if isinstance(thread, list) and thread:
        last = thread[-1]
        # Works for Autogen message objects and dict payloads
        return getattr(last, "content", None) or (last.get("content") if isinstance(last, dict) else str(last))
    return str(thread or "")

# SelectorGroupChat and execution logic
def selector_router(thread, agents = None):
    text = _extract_text_from_thread(thread)
    user_intent = router_intent(text)
    if user_intent['retrieve_medical_summary']:
        return "retrieve_patient_information_agent"
    elif user_intent['search_medical_information']:
        return "lookup_medical_information_agent"
    elif user_intent['manage_medical_history']:
        return "add_medical_records_agent"
    elif user_intent['booking_appointment']:
        return "book_appointment_agent"
    return "lookup_medical_information_agent"

all_agents = [retrieve_patient_information_agent, lookup_medical_information_agent, add_medical_records_agent, book_appointment_agent, group_chat_manager]

selector_group_chat = SelectorGroupChat(
    participants=[retrieve_patient_information_agent, lookup_medical_information_agent, add_medical_records_agent, book_appointment_agent],
    selector_func=partial(selector_router, agents = all_agents),
    allow_repeated_speaker=False,
    termination_condition = TextMentionTermination("TERMINATE"),
    max_turns = 3,
    model_client=model
)

async def run_team_with_retry(team, task: str, retries: int = 4, base_delay: float = 0.4, timeout: float = 45.0):
    """
    Call team.run(task=...) with exponential backoff on OpenAI rate limits.
    Retries: 0.4s, 0.8s, 1.6s, 3.2s (tweak as needed).
    """
    delay = base_delay
    for attempt in range(retries):
        try:
            return await asyncio.wait_for(team.run(task=task), timeout = timeout)
        except RateLimitError as e:
            if attempt == retries - 1:
                raise
            # Optional: log to Streamlit so you can see backoffs
            st.info(f"Rate limit hit; retrying in {delay:.1f}s (attempt {attempt+1}/{retries})…")
            await asyncio.sleep(delay + random.uniform(0, 0.2))
            delay *= 2
        except asyncio.TimeoutError:
            if attempt == retries - 1:
                raise
            st.info(f"Timed out; retrying in {delay:.1f}s (attempt {attempt+1}/{retries})…")
            await asyncio.sleep(delay + random.uniform(0, 0.2))
            delay *= 2        


# =========================
# Streamlit UI
# =========================
st.title("👨‍⚕️ Agentic Healthcare Assistant")

# Keep a persistent chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Repaint prior messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Get user input (guard prevents calls on first load)
user_query = st.chat_input("How can I help you?")
if not user_query:
    st.stop()

# Echo user message and persist it
st.chat_message("user").markdown(user_query)
st.session_state.messages.append({"role": "user", "content": user_query})

# Run the team and SHOW the results
with st.spinner("Thinking..."):
    try:
        result = asyncio.run(run_team_with_retry(selector_group_chat, user_query))
    except RateLimitError:
        st.error("Rate limit from the model. Please try again in a moment.")
        st.stop()
    except Exception as e:
        st.error(f"Oops—something went wrong: {e}")
        st.stop()

    # Render full conversation trace (transient display for this run)
    convo = getattr(result, "messages", None) or getattr(result, "chat_history", None)
    if convo:
        for msg in convo:
            sender = getattr(msg, "name", None) or getattr(msg, "author", "assistant")
            content = getattr(msg, "content", "")
            with st.chat_message(sender if sender in ("user", "assistant") else "assistant"):
                st.write(content)

    # Persist a final assistant reply so it survives future reruns
    final_response = getattr(result, "summary", None) or getattr(result, "final_answer", "")

    if not final_response and convo:
        # Fallback: take the last non-empty assistant/agent message
        for msg in reversed(convo):
            content = getattr(msg, "content", "")
            author  = getattr(msg, "name", None) or getattr(msg, "author", "")
            if content and author != "user":
                final_response = content
                break

    if final_response:
        st.session_state.messages.append({"role": "assistant", "content": final_response})
        st.chat_message("assistant").write(final_response)
