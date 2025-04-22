# app.py - Complete Streamlit Chatbot using Groq API (LCEL Refactor)

# --- Imports ---
import streamlit as st
import os
import sys # Used implicitly by some libraries, good to have
import nltk
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import re
import torch # sentence-transformers uses torch, ensure it's imported
import traceback # For detailed error printing
import faiss # Ensure FAISS is imported for recommend_doctors if needed

# --- LangChain Specific Imports ---
from langchain_groq import ChatGroq
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains import LLMChain # No longer needed for chain definitions with LCEL
from langchain_core.prompts import ChatPromptTemplate

# --- Set Streamlit Page Config (MUST BE THE VERY FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="MedMatch (Groq)", layout="wide")

# --- Inject CSS (Optional - keep if you liked the style) ---
# Example: Title Styling
st.markdown('<h1 style="color: #483D8B;">🩺 MedMatch</h1>', unsafe_allow_html=True)
st.markdown('<h5 style="color: #9370DB;">Fast-Tracking Specialist Access for Smarter, Quicker Care</h5>', unsafe_allow_html=True)
# Example: Chat Bubble Styling (if you were using it)
# st.markdown("""<style>...</style>""", unsafe_allow_html=True)


# --- Download necessary NLTK resources (if not already downloaded) ---
# Streamlit Cloud typically requires specifying download location
NLTK_DATA_PATH = os.path.join(os.path.expanduser("~"), "nltk_data")
if not os.path.exists(os.path.join(NLTK_DATA_PATH, "tokenizers", "punkt")):
    try:
        # Append path only if it's not already there to avoid duplicates
        if NLTK_DATA_PATH not in nltk.data.path:
            nltk.data.path.append(NLTK_DATA_PATH)
        nltk.download('punkt', download_dir=NLTK_DATA_PATH)
        print(f"NLTK 'punkt' downloaded to {NLTK_DATA_PATH}")
    except Exception as e:
        st.error(f"Failed to download NLTK 'punkt': {e}. App may not function correctly.")
        print(f"Error downloading NLTK: {e}")


# --- Configuration ---
DATA_FOLDER = "data"
DB_FAISS_PATH = "vectorstore/db_faiss"
ICD_CSV_PATH = "ICD10-Disease-Mapping.csv"
ICD_CACHE_PATH = "icd_embeddings.pkl"
DOCTOR_PROFILES_PATH = "doctor_profiles_all_specialties.csv"
PATIENT_CASES_PATH = "merged_reviews_new.csv"
SPECIALIST_LIST_PATH = "specialist_categories_list.txt"

GROQ_MODEL = "llama-3.1-8b-instant" # Standard Groq Llama3 70B identifier
GROQ_TEMPERATURE = 1        # Set temperature for more deterministic output

CONVERSATION_TURN_LIMIT = 3
RAG_K = 5

# --- Global Components (Loaded once per session) ---

@st.cache_resource # Cache heavy resources like models, indexes, dataframes
def load_global_components():
    """Loads all necessary models, data, and indices."""
    print("Loading global components...")
    components_loaded = {} # Dictionary to hold components

    # --- Get Groq API Key Securely ---
    groq_api_key = None
    llm_error = None
    try:
        groq_api_key = st.secrets.get("GROQ_API_KEY")
        if not groq_api_key: groq_api_key = os.environ.get("GROQ_API_KEY") # Fallback only if needed

        if not groq_api_key:
            llm_error = "GROQ_API_KEY not found. Please set it in Streamlit secrets."
            # Avoid st.error here as it halts execution too early during caching
            print(f"ERROR: {llm_error}")
        else:
            print("GROQ_API_KEY loaded successfully.")

    except Exception as e:
        llm_error = f"Error accessing secrets: {e}. Make sure GROQ_API_KEY is set."
        print(f"ERROR: {llm_error}")

    # --- Load Components (PDFs, Embeddings, FAISS, ICD, Doctors, Specialists) ---
    # 1. Load PDF Documents
    documents = []
    if os.path.exists(DATA_FOLDER):
        try:
            loader = DirectoryLoader(path=DATA_FOLDER, glob="*.pdf", loader_cls=PyPDFLoader)
            documents = loader.load()
            print(f"Loaded {len(documents)} PDF documents from {DATA_FOLDER}.")
        except Exception as e: print(f"Error loading PDF: {e}"); documents = []
    else: print(f"Data folder not found: {DATA_FOLDER}"); documents = []
    components_loaded["documents"] = documents

    # 2. Initialize Embedding Model
    embeddings = None
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("Embedding model initialized.")
    except Exception as e: print(f"Error initializing embedding model: {e}")
    components_loaded["embeddings"] = embeddings

    # 3. Load or Build FAISS Vectorstore
    vectorstore = None
    if embeddings:
        if os.path.exists(DB_FAISS_PATH):
            try:
                print(f"Loading FAISS vectorstore from cache: {DB_FAISS_PATH}")
                vectorstore = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
                print("FAISS vectorstore loaded.")
            except Exception as e: print(f"Error loading vectorstore: {e}. Attempting rebuild...")
        if vectorstore is None: # Rebuild if loading failed or cache miss
            if documents:
                 try:
                     print("Building FAISS vectorstore from documents...")
                     splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ". "], chunk_size=500, chunk_overlap=50)
                     doc_chunks = splitter.split_documents(documents)
                     print(f"Split into {len(doc_chunks)} chunks.")
                     vectorstore = FAISS.from_documents(doc_chunks, embedding=embeddings)
                     os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
                     vectorstore.save_local(DB_FAISS_PATH)
                     print("FAISS vectorstore rebuilt and saved.")
                 except Exception as e_build: print(f"Error building FAISS: {e_build}")
            else: print("Cannot build FAISS: No documents loaded.")
    else: print("Skipping FAISS: embedding model failed.")
    components_loaded["vectorstore"] = vectorstore

    # 4. Load and Process ICD-10 Mapping & Embeddings
    icd_codes, icd_embeddings = [], None
    if os.path.exists(ICD_CSV_PATH) and embeddings:
        try:
            icd_df = pd.read_csv(ICD_CSV_PATH, dtype=str)
            icd_df.columns = [col.strip(" '\"") for col in icd_df.columns]
            if "ICD-10-CM CODE" in icd_df.columns and "ICD-10-CM CODE DESCRIPTION" in icd_df.columns:
                icd_df = icd_df[["ICD-10-CM CODE", "ICD-10-CM CODE DESCRIPTION"]].dropna().copy() # Ensure clean data & copy
                icd_df.loc[:, "ICD-10-CM CODE"] = icd_df["ICD-10-CM CODE"].str.strip(" '\"")
                icd_df.loc[:, "ICD-10-CM CODE DESCRIPTION"] = icd_df["ICD-10-CM CODE DESCRIPTION"].str.strip(" '\"").str.lower()
                icd_codes = icd_df["ICD-10-CM CODE"].tolist()
                icd_descriptions = icd_df["ICD-10-CM CODE DESCRIPTION"].tolist()
                print(f"Loaded {len(icd_df)} valid ICD10 rows.")

                cache_valid = False
                if os.path.exists(ICD_CACHE_PATH) and os.path.getsize(ICD_CACHE_PATH) > 0:
                    try:
                        print("Loading ICD embeddings from cache...")
                        with open(ICD_CACHE_PATH, "rb") as f: cached_data = pickle.load(f)
                        if isinstance(cached_data, dict) and 'embeddings' in cached_data and 'descriptions' in cached_data:
                            if cached_data['descriptions'] == icd_descriptions:
                                icd_embeddings = cached_data['embeddings']
                                if isinstance(icd_embeddings, np.ndarray) and icd_embeddings.shape[0] == len(icd_descriptions):
                                     print("ICD embeddings loaded and validated from cache.")
                                     cache_valid = True
                                else: print("ICD Cache shape mismatch. Recomputing...")
                            else: print("ICD Cache descriptions mismatch. Recomputing...")
                        else: print("ICD Cache format invalid. Recomputing...")
                    except Exception as e: print(f"Error loading ICD cache: {e}. Recomputing...")

                if not cache_valid:
                    print("Computing ICD embeddings...")
                    try:
                        icd_embeddings_list = embeddings.embed_documents(icd_descriptions)
                        icd_embeddings = np.array(icd_embeddings_list).astype('float32')
                        print("ICD embeddings computed.")
                        try:
                             with open(ICD_CACHE_PATH, "wb") as f: pickle.dump({'embeddings': icd_embeddings, 'descriptions': icd_descriptions}, f)
                             print("ICD embeddings cached.")
                        except Exception as e: print(f"Warning: Could not save ICD cache: {e}")
                    except Exception as e: print(f"Error computing ICD embeddings: {e}")
            else: print("Required ICD columns not found.")
        except Exception as e: print(f"Error processing ICD CSV: {e}")
    else: print("ICD CSV not found or embedding model missing.")
    components_loaded["icd_codes"] = icd_codes
    components_loaded["icd_embeddings"] = icd_embeddings

    # 5. Load Doctor Profiles and Patient Cases
    doctor_df, cases_df = None, None
    try:
        if os.path.exists(DOCTOR_PROFILES_PATH): doctor_df = pd.read_csv(DOCTOR_PROFILES_PATH)
        if os.path.exists(PATIENT_CASES_PATH): cases_df = pd.read_csv(PATIENT_CASES_PATH)
        if doctor_df is not None and cases_df is not None:
             print(f"Loaded {len(doctor_df)} doctors, {len(cases_df)} cases.")
        else: print("Doctor/Cases files missing.")
    except Exception as e: print(f"Error loading Doctor/Cases CSV: {e}")
    components_loaded["doctor_df"] = doctor_df
    components_loaded["cases_df"] = cases_df

    # 6. Load Specialist Categories List
    specialist_categories_list = []
    try:
        if os.path.exists(SPECIALIST_LIST_PATH):
            with open(SPECIALIST_LIST_PATH, "r") as f:
                specialist_categories_list = [line.strip() for line in f.readlines() if line.strip()]
            print(f"Loaded {len(specialist_categories_list)} specialist categories.")
        else: print("Specialist list file not found.")
    except Exception as e: print(f"Error loading specialist list: {e}")
    components_loaded["specialist_categories_list"] = specialist_categories_list

    # 7. Initialize Groq LLM and Chains using LCEL
    llm = None
    followup_chain, final_chain, relevance_check_chain = None, None, None

    if llm_error is None and groq_api_key: # Proceed only if API key is loaded
        try:
            print(f"Initializing Groq model: {GROQ_MODEL} with temp={GROQ_TEMPERATURE}")
            llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name=GROQ_MODEL,
                temperature=GROQ_TEMPERATURE,
            )

            # Define Prompts using ChatPromptTemplate
            # Follow-up Prompt
            followup_prompt_template_str = """System: You are a compassionate and experienced doctor speaking to a patient via chat. Ask a clear, concise follow-up question to gather additional details about their condition based on the latest input. Keep the question simple and focused.
User: {symptoms}
Assistant:"""
            followup_prompt = ChatPromptTemplate.from_template(followup_prompt_template_str)

            # Final Prompt
            final_prompt_template_str = """System: You are a knowledgeable medical assistant. A patient has provided the following combined symptom information: {symptoms}.
Below is reference material about medical specialties and conditions: {context}.
Relevant ICD10 codes: {icd_codes}.
Based *only* on the patient's symptoms and the provided reference material, which type of medical specialist should the patient consult? Provide a concise answer stating the specialist category (e.g., 'Cardiologist') followed by a brief explanation (1-2 sentences) *derived from the context*. If unsure or insufficient info, recommend seeing a 'Primary Care Provider'. Do not include doctor names or contact info.
Assistant:"""
            final_prompt = ChatPromptTemplate.from_template(final_prompt_template_str)

            # Relevance Check Prompt
            relevance_check_prompt_template_str = """System: Analyze the following user conversation history: --- {conversation} ---
Is this conversation primarily focused on describing medical symptoms or seeking medical advice/information related to health conditions? Respond with only the word 'YES' or 'NO'.
Assistant:"""
            relevance_check_prompt = ChatPromptTemplate.from_template(relevance_check_prompt_template_str)

            # Create chains using LCEL (| pipe operator)
            followup_chain = followup_prompt | llm
            final_chain = final_prompt | llm
            relevance_check_chain = relevance_check_prompt | llm

            print("Groq LLM and LangChain LCEL chains initialized.")

        except Exception as e:
            llm_error = f"Error initializing Groq model '{GROQ_MODEL}' or chains: {e}"
            # Avoid st.error here as it halts execution too early during caching
            print(f"ERROR: {llm_error}")
            traceback.print_exc()
            llm, followup_chain, final_chain, relevance_check_chain = None, None, None, None
    else:
         # Store the LLM error if key wasn't found initially
         components_loaded["llm_init_error"] = llm_error

    # Add LLM components to the dictionary
    components_loaded["llm"] = llm
    components_loaded["followup_chain"] = followup_chain
    components_loaded["final_chain"] = final_chain
    components_loaded["relevance_check_chain"] = relevance_check_chain

    # Return all loaded components
    return components_loaded

# --- Load Components ---
components = load_global_components()
embeddings = components.get("embeddings")
vectorstore = components.get("vectorstore")
icd_codes = components.get("icd_codes", [])
icd_embeddings = components.get("icd_embeddings")
doctor_df = components.get("doctor_df")
cases_df = components.get("cases_df")
specialist_categories_list = components.get("specialist_categories_list", [])
llm = components.get("llm")
followup_chain = components.get("followup_chain")
final_chain = components.get("final_chain")
relevance_check_chain = components.get("relevance_check_chain")
llm_init_error = components.get("llm_init_error") # Check for error during loading

# Display LLM init error prominently if it occurred during load
if llm_init_error:
    st.error(f"LLM Initialization Failed: {llm_init_error}")


# --- Helper Functions ---
def match_icd_codes_dense(query, top_n=3, sim_threshold=0.2, icd_codes=None, icd_embeddings=None, embeddings_model=None):
    if not query or icd_embeddings is None or embeddings_model is None or not icd_codes: return ""
    try:
        query_emb_list = embeddings_model.embed_query(query)
        query_emb = np.array([query_emb_list]).astype('float32')
        sims = cosine_similarity(icd_embeddings, query_emb).flatten()
        sorted_indices = np.argsort(sims)[::-1]
        matched = []
        for idx in sorted_indices[:top_n]:
            # Ensure index is valid before accessing sims and icd_codes
            if idx < len(sims) and sims[idx] > sim_threshold and idx < len(icd_codes):
                 matched.append(icd_codes[idx])
        return "; ".join(matched)
    except Exception as e: print(f"Error matching ICD: {e}"); traceback.print_exc(); return ""

def retrieve_context(query, vectorstore=None, k=5):
    if vectorstore is None: return "RAG vector store not available.", []
    try:
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
        retrieved_docs = retriever.invoke(query) # Use invoke for newer LCEL compatibility
        context = "\n---\n".join([doc.page_content for doc in retrieved_docs])
        return context.strip(), retrieved_docs
    except Exception as e: print(f"Error RAG retrieve: {e}"); traceback.print_exc(); return f"Error retrieving context: {e}", []

def extract_specialist(recommendation_text, specialists_list):
    if not recommendation_text or not specialists_list: return None
    recommendation_lower = recommendation_text.lower()
    # Sort list by length descending to match longer names first
    specialists_list_sorted = sorted(specialists_list, key=len, reverse=True)
    for specialist in specialists_list_sorted:
        # Use regex with word boundaries for more accurate matching
        if re.search(r'\b' + re.escape(specialist.lower()) + r'\b', recommendation_lower):
             return specialist # Return the original case specialist name from the list
    return None

def recommend_doctors(category, symptoms, doctor_df=None, cases_df=None, embeddings_model=None):
    if doctor_df is None or cases_df is None or embeddings_model is None or not category or not symptoms: return None
    try:
        # Filtering
        filtered_doctor_df = doctor_df[doctor_df["Specialty"].str.lower() == category.lower()].copy()
        filtered_cases_df = cases_df[cases_df["Specialty"].str.lower() == category.lower()].copy()
        if filtered_cases_df.empty: print(f"Debug: No cases found for {category}"); return pd.DataFrame()
        required_case_cols = ["Symptom Description", "Doctor ID", "Patient Feedback Rating"]
        if not all(col in filtered_cases_df.columns for col in required_case_cols): print("Debug: Cases missing required columns"); return pd.DataFrame()
        filtered_cases_df.dropna(subset=required_case_cols, inplace=True)
        if filtered_cases_df.empty: print(f"Debug: No complete cases found for {category}"); return pd.DataFrame()

        # Embeddings & FAISS
        symptom_texts = filtered_cases_df["Symptom Description"].tolist()
        print(f"Generating embeddings for {len(symptom_texts)} cases in {category}...")
        embeddings_list = embeddings_model.embed_documents(symptom_texts)
        embeddings_np = np.array(embeddings_list).astype("float32")
        if embeddings_np.size == 0: print("Debug: Failed to generate case embeddings."); return pd.DataFrame() # Check if embeddings are empty
        d = embeddings_np.shape[1]
        temp_index = faiss.IndexFlatL2(d); temp_index.add(embeddings_np)
        print(f"Temp FAISS index built: {temp_index.ntotal} vectors.")

        # Case Details Map
        case_details_map = [{'Doctor ID': filtered_cases_df.iloc[i]["Doctor ID"],
                           'Symptom': filtered_cases_df.iloc[i]["Symptom Description"],
                           'Rating': filtered_cases_df.iloc[i]["Patient Feedback Rating"]}
                          for i in range(len(filtered_cases_df))]

        # User Query Embedding & Search
        print(f"Embedding user symptoms for doctor search: '{symptoms[:100]}...'")
        query_embedding_list = embeddings_model.embed_query(symptoms)
        query_embedding = np.array([query_embedding_list]).astype("float32")
        top_k_cases = 10; k_to_search = min(top_k_cases, temp_index.ntotal)
        if k_to_search == 0: print("Debug: Nothing to search in index"); return pd.DataFrame()
        print(f"Starting FAISS search for top {k_to_search} cases...")
        distances, indices = temp_index.search(query_embedding, k_to_search)

        # Gather Similar Cases & Scores
        similar_cases_data = []
        if indices.size > 0 and indices[0][0] != -1:
            for i_idx in indices[0]:
                if i_idx < 0 or i_idx >= len(case_details_map): continue # Skip invalid index
                case_info = case_details_map[i_idx]
                case_embedding = embeddings_np[i_idx]
                sim_score = cosine_similarity(query_embedding, [case_embedding])[0][0]
                similar_cases_data.append({ 'Doctor ID': case_info["Doctor ID"], 'Symptom': case_info["Symptom"],
                                            'Rating': case_info["Rating"], 'Similarity Score': round(float(sim_score), 4)})
        if not similar_cases_data: print("Debug: No similar cases found after processing FAISS results"); return pd.DataFrame()
        similar_cases_df_result = pd.DataFrame(similar_cases_data)

        # Aggregate, Merge, Rank
        similar_cases_df_result.dropna(subset=["Doctor ID", "Rating", "Similarity Score"], inplace=True)
        if similar_cases_df_result.empty: print("Debug: No cases left after dropping NaNs"); return pd.DataFrame()
        doctor_scores_from_similar = similar_cases_df_result.groupby("Doctor ID").agg(
            avg_rating_from_similar=('Rating', 'mean'),
            num_similar_cases=('Doctor ID', 'count'),
            max_similarity_score=('Similarity Score', 'max')).reset_index()
        if "Doctor ID" not in filtered_doctor_df.columns: print("Error: Doctor ID missing in profiles df"); return pd.DataFrame()
        doctor_scores_from_similar['Doctor ID'] = doctor_scores_from_similar['Doctor ID'].astype(str)
        filtered_doctor_df['Doctor ID'] = filtered_doctor_df['Doctor ID'].astype(str)
        recommended_doctors = pd.merge(doctor_scores_from_similar, filtered_doctor_df, on="Doctor ID", how="left")
        sort_cols = ["max_similarity_score", "avg_rating_from_similar", "num_similar_cases"]
        sort_cols_present = [col for col in sort_cols if col in recommended_doctors.columns]
        if sort_cols_present: recommended_doctors = recommended_doctors.sort_values(by=sort_cols_present, ascending=[False] * len(sort_cols_present))

        # Select Top N & Clean
        top_n_doctors = 5
        final_recommendation_df = recommended_doctors.head(top_n_doctors).copy()
        final_recommendation_df.rename(columns={'avg_rating_from_similar': 'Avg Rating (Similar Cases)', 'num_similar_cases': 'Similar Cases Found', 'max_similarity_score': 'Max Similarity Score'}, inplace=True)
        display_cols = ["Doctor ID", "Name", "Specialty", "Avg Rating (Similar Cases)", "Max Similarity Score", "Similar Cases Found", "Years of Experience", "Affiliation"]
        for col in display_cols:
            if col not in final_recommendation_df.columns: final_recommendation_df[col] = None
            # Robust Type Handling
            if col in final_recommendation_df.columns:
                try:
                    if col in ['Doctor ID', 'Name', 'Specialty', 'Affiliation']:
                        final_recommendation_df[col] = final_recommendation_df[col].apply(lambda x: str(x) if pd.notna(x) else "").astype(str)
                    elif col in ['Avg Rating (Similar Cases)', 'Max Similarity Score']:
                        final_recommendation_df[col] = pd.to_numeric(final_recommendation_df[col], errors='coerce').fillna(0.0).round(2)
                    elif col in ['Similar Cases Found', 'Years of Experience']:
                         # Convert to numeric first to handle potential non-int strings like 'Unknown' or '', then to int
                         final_recommendation_df[col] = pd.to_numeric(final_recommendation_df[col], errors='coerce').fillna(0).astype(int)
                except Exception as type_e:
                    print(f"Warning: Type conversion failed for column '{col}': {type_e}")
                    # Apply safe defaults if conversion fails
                    if col in ['Doctor ID', 'Name', 'Specialty', 'Affiliation']: final_recommendation_df[col] = ""
                    else: final_recommendation_df[col] = 0

        print("Returning final recommendation DataFrame.")
        return final_recommendation_df
    except Exception as e: print(f"Error recommend doctors: {e}"); traceback.print_exc(); return None


# --- Streamlit UI ---
st.warning("Disclaimer: This is a prototype using AI and data from public/synthetic sources. It is not a substitute for professional medical advice. Always consult with a qualified healthcare professional for any health concerns.")

# --- Initialize Session State ---
if 'messages' not in st.session_state: st.session_state.messages = []
if 'chat_history_for_llm' not in st.session_state: # Use a separate history for LLM context if needed
     # Define system_prompt string here
    system_prompt_str = """System: You are a helpful and compassionate AI medical assistant chatbot... (Your full system prompt)"""
    st.session_state.chat_history_for_llm = [{"role": "system", "content": system_prompt_str}]
    # Initialize display messages if chat_history is being initialized
    if not st.session_state.messages:
         initial_greeting = "Hello! Please tell me about your symptoms so I can help you figure out what type of doctor you might need."
         st.session_state.messages.append({"role": "assistant", "content": initial_greeting})
         # Optionally add greeting to LLM history too if system prompt allows assistant start
         # st.session_state.chat_history_for_llm.append({"role": "assistant", "content": initial_greeting})

if 'symptom_history' not in st.session_state: st.session_state.symptom_history = "" # Accumulated raw text
if 'user_turn_count' not in st.session_state: st.session_state.user_turn_count = 0
if 'recommendation_made' not in st.session_state: st.session_state.recommendation_made = False
if 'recommended_specialist' not in st.session_state: st.session_state.recommended_specialist = None
if 'final_recommendation_df' not in st.session_state: st.session_state.final_recommendation_df = None
if 'no_doctors_found' not in st.session_state: st.session_state.no_doctors_found = None
if 'rag_context_to_show' not in st.session_state: st.session_state.rag_context_to_show = None
if 'icd_codes_to_show' not in st.session_state: st.session_state.icd_codes_to_show = None


# --- Display Chat Messages ---
chat_container = st.container(height=400, border=True)
with chat_container:
    # Display only 'user' and 'assistant' roles from st.session_state.messages
    for message in st.session_state.messages:
        if message["role"] in ["user", "assistant"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

# --- Chat Input ---
input_disabled = st.session_state.recommendation_made or (llm is None) # Disable if LLM failed
prompt = st.chat_input("Describe your symptoms...", disabled=input_disabled, key="chat_input")

# --- Handle User Input ---
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    # st.session_state.chat_history_for_llm.append({"role": "user", "content": prompt}) # Add to LLM history if using it
    st.session_state.symptom_history += f" User: {prompt}\n"
    st.session_state.user_turn_count += 1
    with chat_container: # Display immediately
        with st.chat_message("user"): st.markdown(prompt)
    # Clear previous results
    if 'final_recommendation_df' in st.session_state: del st.session_state.final_recommendation_df
    if 'no_doctors_found' in st.session_state: del st.session_state.no_doctors_found
    if 'rag_context_to_show' in st.session_state: del st.session_state.rag_context_to_show
    if 'icd_codes_to_show' in st.session_state: del st.session_state.icd_codes_to_show
    st.rerun()


# --- AI Response Logic ---
# Check if the last message was from the user and recommendation hasn't been made
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user" and not st.session_state.recommendation_made:

     # Check if core LLM chains loaded before proceeding
     if llm is None or followup_chain is None or final_chain is None:
         with chat_container:
              with st.chat_message("assistant"):
                   # Check for specific init error message
                   error_msg = components.get("llm_init_error", "AI service core components not available. Please check API key and model setup.")
                   st.error(error_msg)
                   # Avoid adding duplicate error messages to history
                   if not st.session_state.messages or st.session_state.messages[-1].get("content") != error_msg:
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    

         # --- Turn Limit Check ---
     elif st.session_state.user_turn_count >= CONVERSATION_TURN_LIMIT:
         # --- Final Turn Logic ---
         with chat_container:
             with st.chat_message("assistant"):
                 with st.spinner("Analyzing conversation..."):
                     final_response_display = None # For displaying in bubble
                     ai_response_content = ""    # For specialist extraction
                     try:
                         # Relevance Check
                         is_relevant = False
                         if relevance_check_chain:
                             try:
                                 relevance_output_msg = relevance_check_chain.invoke(
                                     {"conversation": st.session_state.symptom_history.strip()} )
                                 relevance_result = relevance_output_msg.content.strip().upper()
                                 print(f"DEBUG Relevance: {relevance_result}")
                                 if relevance_result == 'YES': is_relevant = True
                             except Exception as rel_e:
                                 print(f"Relevance check error: {rel_e}. Assuming relevant.")
                                 traceback.print_exc()
                                 is_relevant = True # Fail safe
                         else:
                             print("Warn: Relevance chain missing. Assuming relevant.")
                             is_relevant = True

                         if is_relevant:
                             print("DEBUG: Conversation relevant. Proceeding with analysis...")
                             st.spinner("Analyzing relevant details...")
                             # RAG & ICD
                             rag_context, _ = retrieve_context(st.session_state.symptom_history.strip(), vectorstore, RAG_K)
                             st.session_state.rag_context_to_show = rag_context # Store even if error occurs later
                             query_icd_codes_str = match_icd_codes_dense(st.session_state.symptom_history.strip(), 5, 0.2, icd_codes, icd_embeddings, embeddings)
                             if not query_icd_codes_str: query_icd_codes_str = "None found."
                             st.session_state.icd_codes_to_show = query_icd_codes_str # Store even if error occurs later

                             # Final LLM Call for Specialist
                             if final_chain:
                                 print("DEBUG: Calling final_chain...")
                                 ai_response_msg = final_chain.invoke({
                                     "symptoms": st.session_state.symptom_history.strip(),
                                     "context": rag_context, "icd_codes": query_icd_codes_str })
                                 ai_response_content = ai_response_msg.content.strip()
                                 print(f"DEBUG: LLM Response Content Received: '{ai_response_content}'") # Show the raw response
                                 # Set the initial display message (can be overwritten if errors occur)
                                 final_response_display = f"**Specialist Recommendation:**\n{ai_response_content}"

                                 # --- DOCTOR RECOMMENDATION GENERATION BLOCK (Revised Logic) ---
                                 print("DEBUG: Checking conditions for doctor recommendation generation...")
                                 # Check if needed dataframes/models/lists are loaded
                                 if doctor_df is not None and cases_df is not None and embeddings is not None and specialist_categories_list:
                                     print("DEBUG: Core data loaded. Attempting specialist extraction...")
                                     # Wrap extraction and recommendation in a try block
                                     try:
                                         extracted_specialist = extract_specialist(ai_response_content, specialist_categories_list)
                                         print(f"DEBUG: Result of extract_specialist: {extracted_specialist}")

                                         # --- CHANGE: Proceed if extraction was successful ---
                                         if extracted_specialist:
                                             st.session_state.recommended_specialist = extracted_specialist
                                             print(f"DEBUG: Calling recommend_doctors for '{extracted_specialist}'...")
                                             recommended_doctors_df = recommend_doctors(extracted_specialist, st.session_state.symptom_history.strip(), doctor_df, cases_df, embeddings)
                                             print(f"DEBUG: recommend_doctors returned: {'DataFrame' if recommended_doctors_df is not None else 'None'}, Empty: {recommended_doctors_df.empty if recommended_doctors_df is not None else 'N/A'}")

                                             # Store results based on what recommend_doctors returned
                                             if recommended_doctors_df is not None and not recommended_doctors_df.empty:
                                                 st.session_state.final_recommendation_df = recommended_doctors_df
                                                 print("DEBUG UI: Stored recommendation DF in session state.")
                                             else:
                                                 st.session_state.no_doctors_found = extracted_specialist
                                                 print(f"DEBUG UI: No doctors found (or empty DF returned) for '{extracted_specialist}', setting flag.")
                                         else: # extract_specialist returned None
                                             st.session_state.no_doctors_found = "Unknown" # Extraction failed
                                             print("DEBUG UI: Could not extract specialist from LLM response, setting 'no_doctors_found' flag.")
                                             # Optionally add info to the display message
                                             # final_response_display += "\n\n(Could not identify specialist for doctor search.)"

                                     except Exception as doc_rec_error:
                                         print(f"ERROR during doctor extraction/recommendation block: {doc_rec_error}")
                                         traceback.print_exc()
                                         # Append error info to the main display message for visibility
                                         error_info = f"\n\n_(Error finding doctors: {doc_rec_error})_"
                                         if final_response_display: final_response_display += error_info
                                         else: final_response_display = error_info # If LLM failed earlier

                                 else: # Essential data/components missing
                                     print("DEBUG: Skipping doctor generation - Missing doctor_df, cases_df, embeddings, or specialist_list.")
                                     # Optionally add info to the display message
                                     # final_response_display += "\n\n_(Doctor recommendation data unavailable.)_"

                             else: # final_chain is None (failed to init)
                                 error_msg = components.get("llm_init_error", "AI specialist recommendation service unavailable.")
                                 final_response_display = f"Error: {error_msg}"
                                 st.error(final_response_display) # Display error in bubble

                         else: # Not Relevant
                             print("DEBUG: Conversation irrelevant.")
                             final_response_display = "It seems our conversation didn't focus on medical symptoms. If you have health concerns, please describe them, and I'll do my best to help. Otherwise, I recommend consulting a qualified healthcare professional for any medical advice."
                             st.session_state.rag_context_to_show = "N/A (Conversation not relevant)"
                             st.session_state.icd_codes_to_show = "N/A (Conversation not relevant)"

                     except Exception as e:
                         print(f"ERROR in final analysis phase: {e}")
                         traceback.print_exc()
                         final_response_display = f"An error occurred during the final analysis process: {e}"
                         st.error(final_response_display) # Show error directly in bubble

                 # --- Update State & Display Final Message ---
                 st.session_state.recommendation_made = True
                 if final_response_display is None: final_response_display = "An error occurred during processing." # Final fallback
                 # Avoid adding duplicate messages
                 if not st.session_state.messages or st.session_state.messages[-1].get("content") != final_response_display:
                     st.session_state.messages.append({"role": "assistant", "content": final_response_display})
                 st.markdown(final_response_display) # Display final message in the chat bubble
                 st.rerun() # Trigger rerun to show results section

     # --- Follow-up Turn Logic ---
     elif followup_chain: # Check if chain exists
             with chat_container:
                 with st.chat_message("assistant"):
                     with st.spinner("Thinking..."):
                         try:
                             response_msg = followup_chain.invoke(
                                 {"symptoms": st.session_state.symptom_history.strip()} )
                             ai_response = response_msg.content.strip()
                             print("Debug: Follow-up question generated.")
                             st.markdown(ai_response)
                             st.session_state.messages.append({"role": "assistant", "content": ai_response})
                         except Exception as e:
                             print(f"Error followup chain: {e}"); traceback.print_exc()
                             error_msg = f"Error generating next question: {e}"
                             st.error(error_msg)
                             st.session_state.messages.append({"role": "assistant", "content": error_msg})
                     st.rerun()
     else: # Handle missing followup chain due to init error
          with chat_container:
              with st.chat_message("assistant"):
                  error_msg = components.get("llm_init_error", "AI followup service not available.")
                  st.error(error_msg)
                  if not st.session_state.messages or st.session_state.messages[-1].get("content") != error_msg:
                       st.session_state.messages.append({"role": "assistant", "content": error_msg})


# --- SEPARATE DISPLAY SECTION ---
if st.session_state.recommendation_made:

    # Doctor Recommendations or "Not Found" message
    if 'final_recommendation_df' in st.session_state and st.session_state.final_recommendation_df is not None:
        df_display = st.session_state.final_recommendation_df
        if not df_display.empty:
            st.markdown("---"); st.markdown("\n**✅ Top Recommended Doctors:**")
            display_cols = ["Doctor ID", "Name", "Specialty", "Avg Rating (Similar Cases)", "Max Similarity Score", "Similar Cases Found", "Years of Experience", "Affiliation"]
            display_cols_present = [col for col in display_cols if col in df_display.columns]
            if display_cols_present: st.dataframe(df_display[display_cols_present], hide_index=True, use_container_width=True)
            else: st.warning("No columns configured for doctor display.")
        # If DF exists but is empty (should be handled by recommend_doctors returning empty df, but double check)
        else:
             specialist_name = st.session_state.get('recommended_specialist', 'the recommended')
             st.info(f"No matching doctor profiles found for **{specialist_name}** specialty based on historical data.")
    elif 'no_doctors_found' in st.session_state:
        specialist_name = st.session_state.no_doctors_found
        if specialist_name and specialist_name != "Unknown": st.info(f"No matching doctor profiles found for **{specialist_name}** specialty based on historical data.")
        elif specialist_name == "Unknown": st.info("Could not search for doctors as specialist was unclear or extraction failed.")

    # Canned response
    canned_response = "Our recommendation has been provided above. Please consult a healthcare professional for further guidance."
    last_message_content = st.session_state.messages[-1].get("content", "") if st.session_state.messages else ""
    if last_message_content != canned_response and not last_message_content.startswith("An error occurred"): # Don't add after error
        with chat_container:
             with st.chat_message("assistant"):
                 st.markdown(canned_response)
                 st.session_state.messages.append({"role": "assistant", "content": canned_response})


# --- Sidebar ---
st.sidebar.header("About")
st.sidebar.info("This is a prototype AI Healthcare Assistant demonstrating symptom-based specialist triage using RAG with Groq and doctor recommendation based on historical case similarity.")
st.sidebar.header("Data Sources")
st.sidebar.markdown(f"- PDFs from '{DATA_FOLDER}'")
st.sidebar.markdown(f"- ICD-10 Mapping from '{ICD_CSV_PATH}'")
st.sidebar.markdown(f"- Sample Doctor/Case data from '{DOCTOR_PROFILES_PATH}' & '{PATIENT_CASES_PATH}'")
st.sidebar.markdown("*(Note: Data sources are for demonstration)*")

# --- Restart Button Logic ---
if st.sidebar.button("Restart Chat"):
    # Keys to reset
    keys_to_reset = [
        'messages', 'chat_history_for_llm', 'symptom_history', 'user_turn_count',
        'recommendation_made', 'recommended_specialist', 'final_recommendation_df',
        'no_doctors_found', 'rag_context_to_show', 'icd_codes_to_show'
    ]
    # Preserve system prompt if it exists
    system_prompt_message = None
    if 'chat_history_for_llm' in st.session_state and st.session_state.chat_history_for_llm:
        if st.session_state.chat_history_for_llm[0].get('role') == 'system':
            system_prompt_message = st.session_state.chat_history_for_llm[0]

    # Clear state
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]

    # Re-initialize necessary state
    st.session_state.messages = []
    st.session_state.chat_history_for_llm = [system_prompt_message] if system_prompt_message else []
    st.session_state.symptom_history = ""
    st.session_state.user_turn_count = 0
    st.session_state.recommendation_made = False

    # Add greeting back if system prompt exists (implies successful LLM init)
    if st.session_state.chat_history_for_llm:
        initial_greeting = "Hello! Please tell me about your symptoms so I can help you figure out what type of doctor you might need."
        st.session_state.messages.append({"role": "assistant", "content": initial_greeting})
        # Optionally add greeting to LLM history too
        # st.session_state.chat_history_for_llm.append({"role": "assistant", "content": initial_greeting})

    st.rerun()
