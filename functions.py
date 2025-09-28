import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from typing import Optional, Dict, Any, List
import nltk
from nltk.stem import WordNetLemmatizer
from key_words import BOOKING_KEY_WORDS,RETRIEVE_KEY_WORDS,SEARCH_KEY_WORDS,HISTORY_KEY_WORDS
import psycopg
from dotenv import load_dotenv
import os
import datetime
from psycopg.rows import dict_row
from Bio import Entrez
import random

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
database_connection = os.getenv("DATABASE_URL")
email = os.getenv("EMAIL")
ncbi_api_key = os.getenv("NCBI_API_KEY")

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