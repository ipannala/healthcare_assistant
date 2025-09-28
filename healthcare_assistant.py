import streamlit as st
import autogen
from autogen.agentchat.contrib.capabilities.tools import Tool, ToolManager
from autogen import ConversableAgent, UserProxyAgent, AssistantAgent, GroupChat, GroupChatManager
from autogen.oai import config_list_openai_aoai
from autogen.agentchat.assistant_agent import AssistantAgent
from autogen.agentchat.user_proxy_agent import UserProxyAgent
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
import os
import psycopg
import datetime
from psycopg.rows import dict_row
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from Bio import Entrez
import json
import logging

# Set up logging to capture agent output
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
database_connection = os.getenv("DATABASE_URL")
email = os.getenv("EMAIL")
ncbi_api_key = os.getenv("NCBI_API_KEY")

# Set up the OpenAI model client
config_list = [
    {
        "model": "gpt-4o",
        "api_key": openai_api_key
    }
]

# Define keywords for intent router
BOOKING_KEY_WORDS = {'book', 'schedule', 'appointment'}
RETRIEVE_KEY_WORDS = {'retrieve', 'recover', 'summarize', 'summary'}
SEARCH_KEY_WORDS = {'search', 'lookup', 'look', 'find'}
HISTORY_KEY_WORDS = {'add', 'update', 'append'}
POS_MAP = {
    "NOUN": "n",
    "VERB": "v",
    "ADJ": "a",
    "ADV": "r"
}

# Functions from the original `functions.py` file
def router_intent(messages: str) -> dict:
    user_intent_dict: Dict[str, Any] = {
        'booking_appointment': False,
        'retrieve_medical_summary': False,
        'search_medical_information': False,
        'manage_medical_history': False
    }
    messages = messages.lower()
    tokenized_words = word_tokenize(messages)
    pos_tagged_words = pos_tag(tokenized_words, tagset='universal')
    wnl = WordNetLemmatizer()
    lemmatized_words = []
    for word, pos in pos_tagged_words:
        wn_pos = POS_MAP.get(pos)
        if wn_pos:
            lemmatized_words.append(wnl.lemmatize(word, pos=wn_pos))
        else:
            lemmatized_words.append(word)
    lemmatized_words_set = set(lemmatized_words)
    if lemmatized_words_set & BOOKING_KEY_WORDS:
        user_intent_dict['booking_appointment'] = True
    if lemmatized_words_set & RETRIEVE_KEY_WORDS:
        user_intent_dict['retrieve_medical_summary'] = True
    if lemmatized_words_set & SEARCH_KEY_WORDS:
        user_intent_dict['search_medical_information'] = True
    if lemmatized_words_set & HISTORY_KEY_WORDS:
        user_intent_dict['manage_medical_history'] = True
    return user_intent_dict

def retrieve_patient_information(patient_id: int) -> Optional[tuple[str, str, str]]:
    conn = psycopg.connect(database_connection)
    cur = conn.cursor(row_factory=dict_row)
    cur.execute("""SELECT condition_names, medications, doctor_notes FROM public."ehr_records" where patient_id = %s""", (patient_id,))
    row = cur.fetchone()
    if row is None:
        cur.close()
        conn.close()
        return None
    condition_information = row['condition_names']
    medication_information = row['medications']
    doctor_notes = row['doctor_notes']
    cur.close()
    conn.close()
    return condition_information, medication_information, doctor_notes

def lookup_medical_information(search_query: str) -> str:
    current_year = datetime.date.today().year
    year_from = current_year - 5
    year_to = current_year
    Entrez.email = email
    handle = Entrez.esearch(db="pubmed", term=search_query, datetype="pdat", mindate=str(year_from), maxdate=str(year_to), retmax=5)
    record = Entrez.read(handle)
    handle.close()
    article_ids = record['IdList']
    id_string = ",".join(article_ids[:5])
    fetch_handle = Entrez.efetch(db="pubmed", id=id_string, retmode="xml")
    articles = Entrez.read(fetch_handle)
    fetch_handle.close()
    string_text = ""
    for article in articles['PubmedArticle']:
        for abstract_text in article['MedlineCitation']['Article']['Abstract']['AbstractText']:
            string_text += str(abstract_text)
            string_text += " "
    return string_text

def add_medical_records(patient_id: int, condition_names: str, medications: str, doctor_notes: str) -> str:
    conn = psycopg.connect(database_connection)
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO public."ehr_records" (patient_id, condition_names, medications, doctor_notes)
           VALUES (%s, %s, %s, %s)
           ON CONFLICT (patient_id) DO UPDATE
           SET condition_names = EXCLUDED.condition_names,
               medications = EXCLUDED.medications,
               doctor_notes = EXCLUDED.doctor_notes
           WHERE public."ehr_records".patient_id = EXCLUDED.patient_id;""",
        (patient_id, condition_names, medications, doctor_notes)
    )
    conn.commit()
    cur.close()
    conn.close()
    return "Patient record updated successfully."

def book_appointment(patient_id: int, doctor_name: str, appointment_time: datetime.datetime) -> str:
    conn = psycopg.connect(database_connection)
    cur = conn.cursor()
    cur.execute(
        """SELECT doctor_id FROM public.doctor_schedule WHERE doctor_name = %s AND specialty = 'Nephrology'""", (doctor_name,)
    )
    doctor_id = cur.fetchone()
    if not doctor_id:
        return "Doctor not found."
    cur.execute(
        """INSERT INTO public.appointments (patient_id, doctor_id, appointment_time) VALUES (%s, %s, %s)""",
        (patient_id, doctor_id, appointment_time)
    )
    conn.commit()
    cur.close()
    conn.close()
    return f"Appointment booked successfully for patient {patient_id} with Dr. {doctor_name} at {appointment_time}"

# Agent definitions
retrieve_patient_information_agent = AssistantAgent(
    name='retrieve_patient_information_agent',
    llm_config={"config_list": config_list},
    system_message=(
        "You are an expert medical assistant.\n"
        "You have access to a tool named retrieve_patient_information.\n"
        "First call the retrieve_patient_information_tool using the patient ID in this format: retrieve_patient_information(patient ID) function to retrieve patient information.\n"
        "Summarize the medical condition, medications patient is using, and the doctor notes."
    )
)

lookup_medical_information_agent = AssistantAgent(
    name="lookup_medical_information_agent",
    llm_config={"config_list": config_list},
    system_message=(
        "You are an expert medical assistant.\n"
        "You have access to a tool named lookup_medical_information.\n"
        "First call the lookup_medical_information_tool using the user query in this format: lookup_medical_information(user query) function to search medical journals about the medical question.\n"
        "Summarize the medical journal abstract text in about 6 - 8 sentences. Ensure the important information from the medical journal abstract text is included in the summary.\n"
    )
)

add_medical_records_agent = AssistantAgent(
    name="add_medical_records_agent",
    llm_config={"config_list": config_list},
    system_message=(
        "You are an expert medical assistant for managing medical records.\n"
        "You have access to a tool named add_medical_records.\n"
        "Retrieve the condition name, medications from the doctor notes.\n"
        "Summarize the doctor notes.\n"
        "Then call the add_medical_records_tool and insert this information in the tool by calling the add_medical_records python function in this format: add_medical_records(patient id, condition names, medications, doctor notes)."
    )
)

book_appointment_agent = AssistantAgent(
    name="book_appointment_agent",
    llm_config={"config_list": config_list},
    system_message=(
        "You are an expert medical assistant for booking appointments.\n"
        "You have access to a tool named book_appointment.\n"
        "Retrieve the patient id, full name of the doctor, and appointment time. Call the book_appointment_tool by calling the book_appointment python function in this format: book_appointment(patient_id, doctor_name, appointment_time)."
    )
)

user_proxy = UserProxyAgent(
    name="User_Proxy",
    human_input_mode="NEVER",
    code_execution_config={"use_docker": False}
)

# Register the tools with the agents
user_proxy.register_for_execution(retrieve_patient_information, name="retrieve_patient_information")
user_proxy.register_for_execution(lookup_medical_information, name="lookup_medical_information")
user_proxy.register_for_execution(add_medical_records, name="add_medical_records")
user_proxy.register_for_execution(book_appointment, name="book_appointment")

# Define the group chat with the selector function
def selector_router(messages: list[Dict], agents: List[ConversableAgent]) -> Optional[ConversableAgent]:
    last_message = messages[-1]['content']
    user_intent = router_intent(last_message)
    if user_intent['retrieve_medical_summary']:
        return retrieve_patient_information_agent
    elif user_intent['search_medical_information']:
        return lookup_medical_information_agent
    elif user_intent['manage_medical_history']:
        return add_medical_records_agent
    elif user_intent['booking_appointment']:
        return book_appointment_agent
    return None

group_chat = GroupChat(
    agents=[
        user_proxy,
        retrieve_patient_information_agent,
        lookup_medical_information_agent,
        add_medical_records_agent,
        book_appointment_agent,
    ],
    messages=[],
    speaker_selection_method=selector_router
)

group_chat_manager = GroupChatManager(
    groupchat=group_chat,
    llm_config={"config_list": config_list}
)

# Streamlit UI
st.title("👨‍⚕️ Agentic Healthcare Assistant")

# Initialize chat history in Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input and initiate chat
if user_query := st.chat_input("How can I help you?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(user_query)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Run the chat and capture the response
    with st.spinner("Thinking..."):
        # The core logic to run the chat
        response = user_proxy.initiate_chat(
            group_chat_manager,
            message=user_query,
            clear_history=True,
            silent=True
        )
        
        # Display the chat history from the result
        for msg in response.chat_history:
            sender_name = msg["name"]
            content = msg["content"]
            
            with st.chat_message(sender_name):
                st.write(content)

        # Store the final response for the chat history
        final_response = response.summary
        st.session_state.messages.append({"role": "assistant", "content": final_response})
        st.experimental_rerun()