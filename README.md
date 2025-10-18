# Healthcare Assistant
A multi-agent healthcare assistant designed to assist healthcare providers with common tasks such as retrieving patient information, searching medical literature, updating patient records, and booking appointments. This assistant is built using a multi-agent framework to route queries to specialized agents. 

## Key Features
* **Intelligent Intent Routing:** Uses NLP (NLTK) and keyword matching to instantly route user queries (e.g "book", "search", "retrieve") to the correct specialized agent
* **Medical Information Search:** Integrates with PubMed via the Entrez API to look up and summarize the latest medical literature on specific conditions
* **Patient Record Management:** Retrieves, adds, and updates patient health records (conditions, medications, doctor notes) in a PostgreSQL database
* **Appointment Scheduling**: Facilitates  booking appointments with doctors by interacting with a scheduling database

## Architecture and Agent Roles

The system is powered by a **Selector Group Chat** that directs a user's request to one of four specialized agents, optimizing the use of tools and model resources. The **`router_intent()`** function acts as the traffic cop, analyzing the user's message using NLP (tokenization and lemmatization) and keyword matching to determine the correct agent.

| Agent Name | Primary Function | Key Tool/Function Called | Intent Keyword Categories |
| :--- | :--- | :--- | :--- |
| **`retrieve_patient_information_agent`** | Summarize existing medical records (conditions, medications, notes) for a given patient ID. | `retrieve_patient_information(patient_id)` | **Retrieve** (`retrieve`, `summarize`, `summary`) |
| **`lookup_medical_information_agent`** | Search PubMed medical journals and summarize relevant abstract texts from the past 5 years. | `lookup_medical_information(search_query)` | **Search** (`search`, `lookup`, `find`) |
| **`add_medical_records_agent`** | Update a patient's EHR with new condition, medication, and doctor notes. | `add_medical_records(patient_id, condition_names, medications, doctor_notes)` | **History** (`add`, `update`, `append`) |
| **`book_appointment_agent`** | Handle the scheduling of a new doctor's appointment. | `book_appointment(patient_id, doctor_name, start_time, end_time, date)` | **Booking** (`book`, `schedule`, `appointment`) |

The router_intent() function acts as the traffic cop, analyzing the user's message using NLP (tokenization and lemmatization with NLTK) and keyword matching to determine the correct agent.

## Getting Started
**Prerequisites**
* Python 3.12+

Follow these steps to set up and run the Healthcare Assistant Agent locally.

### 1. Setup Environment labels
The project requires several API keys and connection strings to operate its various services (OpenAI, PubMed, and PostgreSQL).
Create a file named **`.env`** in the root of your project directory and fill it with your credentials:

```bash
# --- OpenAI API Key ---
# Used by the model client for all agent reasoning and summarization.
OPENAI_API_KEY="sk-..."

# --- NCBI (PubMed) Entrez API ---
# The email address is required by the Bio.Entrez module for PubMed searches.
EMAIL="your.email@example.com"
# NCBI_API_KEY is optional, but recommended for high volume lookups.
NCBI_API_KEY="..." 

# --- PostgreSQL Database Connection ---
# Connection string for the database storing EHR and appointment data.
DATABASE_URL="postgresql://user:password@host:port/database_name"
```
### 2. Database Setup

The agent requires access to a **PostgreSQL** database with the following tables. The `functions.py` file outlines the specific columns needed for the agent tools to operate correctly.

#### A. `ehr_records` Table (for record retrieval and updates)

This table is used by the **`retrieve_patient_information_agent`** and **`add_medical_records_agent`**

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `patient_id` | `INT` | **Primary Key** for the patient; Used for conflict resolution on updates. |
| `condition_names` | `TEXT` | List or summary of the patient's medical conditions |
| `medications` | `TEXT` | List of the patient's current medications |
| `doctor_notes` | `TEXT` | Latest comprehensive doctor's notes/summary |

#### B. `doctor_information`

This table is used by the **`book_appointment_agent`**

| Column Name | Data Type | Description | 
| :--- | :--- | :--- |
| `doctor_id`  | `INT` | **Automatically Generated Primary Key** for the doctor ID
| `full_name`  |   `TEXT` | Full name of the doctor

#### C. `doctor_schedule`

This table is used by the **`book_appointment_agent`**

| Column Name | Data Type | Description | 
| :--- | :--- | :--- |
| `appointment_id` | `INT` | **Automatically Generated Primary Key** appointment ID |
| `doctor_id` | `INT`  | Doctor ID of doctor|
| `patient_id` | `INT`  | Patient ID of patient|
| `start_time` | `TIME`  | Starting time of appointment|
| `end_time` | `TIME`  | Ending time of appointment|
| `date` | `DATE`  | Date of appointment|

### 3. Install Dependencies

The necessary libraries for the multi-agent system, database connection, and medical search functionality must be installed.

You can install all dependencies if you have a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

```bash
streamlit run healthcare_assistant.py
```

## Example Interactions

The assistant uses an intelligent router to direct your query to the correct specialized agent. Here are a few ways to interact with the assistant:

* **Retrieve Summary (Routes to `retrieve_patient_information_agent`):**
    * Example: "Can you retrieve the medical summary for patient 5017?"
* **Search Medical Info (Routes to `lookup_medical_information_agent`):**
    * Example: "What are the latest findings on personalized medicine for heart disease?"
* **Manage History (Routes to `add_medical_records_agent`):**
    * Example: "Update patient 123's records with condition 'Type 2 Diabetes', medications 'Metformin', and doctor notes: 'Patient showed signs of improvement and was prescribed Metformin'."
* **Book Appointment (Routes to `book_appointment_agent`):**
    * Example: "Schedule an appointment for patient 456 with Dr. Smith tomorrow at 10:00 AM."




  










