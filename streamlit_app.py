# app.py - Complete Streamlit Chatbot Interface with RAG and Doctor Recommendation

import streamlit as st
import os
import sys
import nltk
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import re
import torch # sentence-transformers uses torch, ensure it's imported
import traceback # For detailed error printing

# --- Set Streamlit Page Config (MUST BE THE VERY FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="AI Healthcare Assistant", layout="wide")


# --- Download necessary NLTK resources (if not already downloaded) ---
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
except LookupError:
     nltk.download('punkt')

# --- Configuration ---
DATA_FOLDER = "data"  # Folder containing your PDFs.
DB_FAISS_PATH = "vectorstore/db_faiss" # Path to your saved FAISS vectorstore.
ICD_CSV_PATH = "ICD10-Disease-Mapping.csv"
ICD_CACHE_PATH = "icd_embeddings.pkl" # Path to cached ICD embeddings.
DOCTOR_PROFILES_PATH = "doctor_profiles_all_specialties.csv" # Path to doctor profiles.
PATIENT_CASES_PATH = "merged_reviews_new.csv" # Path to patient cases/reviews.
SPECIALIST_LIST_PATH = "specialist_categories_list.txt" # Path to list of specialist names.

OLLAMA_MODEL = "llama3.2:3b"  # Your Ollama model identifier.
CONVERSATION_TURN_LIMIT = 3 # Number of user turns before triggering RAG + Final Recommendation. (e.g., 1 initial + 2 follow-ups)
RAG_K = 5 # Number of top chunks to retrieve from FAISS.

# --- Global Components (Loaded once per session) ---

@st.cache_resource # Cache heavy resources like models, indexes, dataframes
def load_global_components():
    """Loads all necessary models, data, and indices."""
    print("Loading global components...")

    # 1. Load PDF Documents (or indicate if data folder is missing)
    # Correct import based on deprecation warnings
    from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
    documents = []
    if os.path.exists(DATA_FOLDER):
        try:
            loader = DirectoryLoader(
                path=DATA_FOLDER,
                glob="*.pdf",
                loader_cls=PyPDFLoader
            )
            documents = loader.load()
            print(f"Loaded {len(documents)} PDF documents from {DATA_FOLDER}.")
        except Exception as e:
            print(f"Error loading PDF documents from {DATA_FOLDER}: {e}")
            st.error(f"Could not load PDF documents from '{DATA_FOLDER}'. Please ensure the folder exists and contains PDFs.")
            documents = [] # Ensure documents is empty if loading fails
    else:
        print(f"Data folder not found at {DATA_FOLDER}.")
        st.warning(f"Data folder '{DATA_FOLDER}' not found. RAG may not work if documents are missing.")
        documents = []


    # 2. Initialize Embedding Model (for both document chunks and ICD)
    # Correct import based on deprecation warnings
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings = None # Initialize to None
    try:
        # Still use the model_name here
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("Embedding model initialized.")
    except Exception as e:
        print(f"Error initializing embedding model: {e}")
        st.error(f"Could not load embedding model. RAG and ICD matching will fail.")
        embeddings = None # Ensure it's None on failure


    # 3. Load or Build FAISS Vectorstore (for RAG on document chunks)
    # Correct import based on deprecation warnings
    from langchain_community.vectorstores import FAISS
    vectorstore = None # Initialize to None
    if embeddings: # Only proceed if embeddings loaded
        if os.path.exists(DB_FAISS_PATH):
            try:
                print(f"Loading FAISS vectorstore from cache: {DB_FAISS_PATH}")
                vectorstore = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
                print("FAISS vectorstore loaded.")
            except Exception as e:
                print(f"Error loading vectorstore: {e}. Attempting to rebuild...")
                # Rebuild if loading fails
                if documents:
                     try:
                         print("Building FAISS vectorstore from documents...")
                         # Use a recursive splitter before building FAISS if documents loaded
                         from langchain.text_splitter import RecursiveCharacterTextSplitter
                         optimized_splitter = RecursiveCharacterTextSplitter(
                             separators=["\n\n", "\n", ". "],
                             chunk_size=500,       # Adjust as needed
                             chunk_overlap=50
                         )
                         doc_chunks = optimized_splitter.split_documents(documents)
                         print(f"Split documents into {len(doc_chunks)} chunks for indexing.")
                         vectorstore = FAISS.from_documents(doc_chunks, embedding=embeddings)
                         vectorstore.save_local(DB_FAISS_PATH)
                         print("FAISS vectorstore rebuilt and saved.")
                     except Exception as e_build:
                         print(f"Error building FAISS vectorstore: {e_build}")
                         st.error(f"Could not build FAISS vectorstore. RAG will not work.")
                         vectorstore = None
                else:
                    print("Cannot rebuild FAISS vectorstore: No documents loaded.")
                    st.error("Cannot rebuild FAISS vectorstore. RAG will not work.")
                    vectorstore = None
        else:
            print(f"No cached vectorstore found at {DB_FAISS_PATH}. Attempting to build...")
            if documents:
                try:
                    print("Building FAISS vectorstore from documents...")
                    # Use a recursive splitter before building FAISS
                    from langchain.text_splitter import RecursiveCharacterTextSplitter
                    optimized_splitter = RecursiveCharacterTextSplitter(
                        separators=["\n\n", "\n", ". "],
                        chunk_size=500, # Adjust as needed
                        chunk_overlap=50
                    )
                    doc_chunks = optimized_splitter.split_documents(documents)
                    print(f"Split documents into {len(doc_chunks)} chunks for indexing.")
                    vectorstore = FAISS.from_documents(doc_chunks, embedding=embeddings)
                    vectorstore.save_local(DB_FAISS_PATH)
                    print("FAISS vectorstore built and saved.")
                except Exception as e_build:
                    print(f"Error building FAISS vectorstore: {e_build}")
                    st.error(f"Could not build FAISS vectorstore. RAG will not work.")
                    vectorstore = None
            else:
                print("Cannot build FAISS vectorstore: No documents loaded.")
                st.error("Cannot build FAISS vectorstore. RAG will not work.")
                vectorstore = None
    else:
        print("Skipping FAISS loading/building as embedding model failed.")
        st.error("Skipping FAISS loading/building. RAG will not work.")


    # 4. Load and Process ICD-10 Mapping
    icd_codes, icd_descriptions, icd_embeddings = [], [], None # Removed vectorizer here as not used globally
    if os.path.exists(ICD_CSV_PATH):
        try:
            icd_df = pd.read_csv(ICD_CSV_PATH, dtype=str)
            icd_df.columns = [col.strip(" '\"") for col in icd_df.columns] # Clean column names
            if "ICD-10-CM CODE" in icd_df.columns and "ICD-10-CM CODE DESCRIPTION" in icd_df.columns:
                icd_df["ICD-10-CM CODE"] = icd_df["ICD-10-CM CODE"].str.strip(" '\"")
                icd_df["ICD-10-CM CODE DESCRIPTION"] = icd_df["ICD-10-CM CODE DESCRIPTION"].str.strip(" '\"").str.lower()
                icd_codes = icd_df["ICD-10-CM CODE"].tolist()
                icd_descriptions = icd_df["ICD-10-CM CODE DESCRIPTION"].tolist()
                print(f"Loaded ICD10 mapping data: {len(icd_df)} rows.")

                # Try to load cached ICD embeddings
                if os.path.exists(ICD_CACHE_PATH) and os.path.getsize(ICD_CACHE_PATH) > 0:
                    try:
                        print("Attempting to load ICD embeddings from cache...")
                        with open(ICD_CACHE_PATH, "rb") as f:
                            cached_data = pickle.load(f)
                            icd_embeddings = cached_data['embeddings']
                            # Optional: Check if cached descriptions match current ones
                            if 'descriptions' in cached_data and cached_data['descriptions'] == icd_descriptions:
                                print("ICD embeddings loaded from cache.")
                            else:
                                print("Cached ICD descriptions don't match or missing. Recomputing embeddings...")
                                icd_embeddings = None # Force recomputation
                    except (EOFError, pickle.UnpicklingError, KeyError) as e: # Corrected UnpicklingError
                        print(f"ICD cache file corrupted or format incorrect: {e}. Recomputing embeddings...")
                        icd_embeddings = None # Force recomputation
                    except Exception as e:
                        print(f"Unexpected error loading ICD cache: {e}. Recomputing embeddings...")
                        icd_embeddings = None # Force recomputation

                if icd_embeddings is None and embeddings is not None: # Compute if cache failed or missing and embeddings loaded
                    print("Computing ICD embeddings in batch...")
                    # Use embeddings.embed_documents() from the LangChain wrapper
                    try:
                         icd_embeddings_list = embeddings.embed_documents(icd_descriptions)
                         # Convert to numpy array and ensure float32 for calculations
                         icd_embeddings = np.array(icd_embeddings_list).astype('float32')
                         print("ICD embeddings computed.")
                         # Cache the newly computed embeddings and descriptions
                         try:
                             with open(ICD_CACHE_PATH, "wb") as f:
                                 pickle.dump({'embeddings': icd_embeddings, 'descriptions': icd_descriptions}, f)
                             print("ICD embeddings cached.")
                         except Exception as e:
                              print(f"Warning: Could not save ICD embeddings cache: {e}")
                    except Exception as e:
                         print(f"Error computing ICD embeddings: {e}")
                         st.error(f"Error computing ICD embeddings: {e}. ICD matching will not work.")
                         icd_embeddings = None # Ensure None on computation failure

            else:
                print(f"Required columns not found in ICD CSV: {ICD_CSV_PATH}. Skipping ICD mapping.")
                st.warning(f"ICD mapping skipped. File '{ICD_CSV_PATH}' missing or format incorrect.")
        except FileNotFoundError:
            print(f"ICD10 mapping file not found: {ICD_CSV_PATH}. Skipping ICD mapping.")
            st.warning(f"ICD mapping skipped. File '{ICD_CSV_PATH}' not found.")
        except Exception as e:
            print(f"Error loading or processing ICD10 mapping CSV: {e}. Skipping ICD mapping.")
            st.error(f"Error loading or processing ICD10 mapping CSV: {e}. Skipping ICD mapping.")

    else:
        print(f"ICD10 mapping file not found: {ICD_CSV_PATH}. Skipping ICD mapping.")
        st.warning(f"ICD mapping skipped. File '{ICD_CSV_PATH}' not found.")


    # 5. Load Doctor Profiles and Patient Cases (for Doctor Recommendation)
    doctor_df, cases_df = None, None
    if os.path.exists(DOCTOR_PROFILES_PATH) and os.path.exists(PATIENT_CASES_PATH):
        try:
            doctor_df = pd.read_csv(DOCTOR_PROFILES_PATH)
            cases_df = pd.read_csv(PATIENT_CASES_PATH)
            print(f"Loaded {len(doctor_df)} doctor profiles and {len(cases_df)} patient cases.")
        except FileNotFoundError as e:
             print(f"Doctor/Cases file not found: {e}. Doctor recommendation will not work.")
             st.warning(f"Doctor recommendation skipped. File '{e.filename}' not found.")
        except Exception as e:
            print(f"Error loading Doctor/Cases data: {e}. Doctor recommendation will not work.")
            st.error(f"Error loading Doctor/Cases data: {e}. Doctor recommendation will not work.")
    else:
        print(f"Doctor/Cases files not found. Doctor recommendation will not work.")
        st.warning(f"Doctor recommendation skipped. Files '{DOCTOR_PROFILES_PATH}' or '{PATIENT_CASES_PATH}' not found.")

    # 6. Load Specialist Categories List (for extracting specialist name from LLM output)
    specialist_categories_list = []
    if os.path.exists(SPECIALIST_LIST_PATH):
        try:
            with open(SPECIALIST_LIST_PATH, "r") as f:
                specialist_categories_list = [line.strip() for line in f.readlines() if line.strip()]
            print(f"Loaded {len(specialist_categories_list)} specialist categories list.")
        except FileNotFoundError:
            print(f"Specialist list file not found: {SPECIALIST_LIST_PATH}. Specialist extraction might be less accurate.")
            st.warning(f"Specialist list missing: {SPECIALIST_LIST_PATH}. Specialist extraction might be less accurate.")
        except Exception as e:
            print(f"Error loading specialist list: {e}. Specialist extraction might be less accurate.")
            st.warning(f"Error loading specialist list: {e}. Specialist extraction might be less accurate.")
    else:
        print(f"Specialist list file not found: {SPECIALIST_LIST_PATH}. Specialist extraction might be less accurate.")
        st.warning(f"Specialist list missing: {SPECIALIST_LIST_PATH}. Specialist extraction might be less accurate.")


    # 7. Initialize Ollama LLM and LangChain Chains
    # Correct import based on deprecation warnings
    from langchain_community.chat_models import ChatOllama
    from langchain.chains import LLMChain # Keep for now as in your code, though deprecated
    from langchain.prompts import PromptTemplate # Keep for now as in your code

    llm = None
    followup_chain, final_chain = None, None

    # Define Prompts
    followup_prompt_template = """
    You are a compassionate and experienced doctor speaking to a patient via chat.
    The patient describes their symptoms as: "{symptoms}".
    Ask a clear, concise follow-up question to gather additional details about their condition.
    Keep the question simple and focused.
    """
    followup_prompt = PromptTemplate(
        template=followup_prompt_template,
        input_variables=["symptoms"]
    )

    final_prompt_template = """
    You are a knowledgeable medical assistant.
    A patient has provided the following combined symptom information across our conversation:
    {symptoms}

    Below is relevant reference material about medical specialties and the conditions they treat:
    {context}

    Additionally, based on analysis of the symptoms, here are relevant medical codes (ICD10):
    {icd_codes}

    Based *only* on the patient's symptoms and the provided reference material, which type of medical specialist should the patient consult?
    Provide a concise answer stating the specialist category (e.g., 'Cardiologist', 'Dermatologist', 'Neurologist', 'Primary Care Physician', etc.).
    Follow the specialist name with a brief explanation (1-2 sentences) *derived from the context* about what that specialist does or why they are relevant to the symptoms.
    If the provided information is insufficient to confidently recommend a specific specialist, state that you cannot make a definitive recommendation and advise seeing a general physician (Primary Care Provider) first.
    Do not include contact information or specific doctor names.
    """

    final_prompt = PromptTemplate(
        template=final_prompt_template,
        input_variables=["symptoms", "context", "icd_codes"]
    )


    try:
        print(f"Initializing Ollama model: {OLLAMA_MODEL}")
        llm = ChatOllama(model=OLLAMA_MODEL)

        # Use .bind() or .with_config() for newer LangChain syntax if you move away from LLMChain
        # For now, keeping LLMChain as in your code, though it's deprecated
        followup_chain = LLMChain(llm=llm, prompt=followup_prompt)
        final_chain = LLMChain(llm=llm, prompt=final_prompt)
        print("Ollama model and LangChain chains initialized.")
    except Exception as e:
        print(f"Error initializing the Ollama model '{OLLAMA_MODEL}' or LangChain chains: {e}")
        st.error(f"Could not initialize Ollama model '{OLLAMA_MODEL}'. Ensure it's running and pulled. AI conversation will not work.")
        llm, followup_chain, final_chain = None, None, None # Ensure they are None on failure


    # Return all loaded components
    return {
        "embeddings": embeddings, # Keep the embedding object for ICD matching and Doctor Recs
        "vectorstore": vectorstore,
        "icd_codes": icd_codes,
        "icd_embeddings": icd_embeddings, # Keep the embeddings
        "doctor_df": doctor_df,
        "cases_df": cases_df,
        "specialist_categories_list": specialist_categories_list,
        "llm": llm, # Keep the LLM instance
        "followup_chain": followup_chain, # Keep the chain instance
        "final_chain": final_chain, # Keep the chain instance
    }

# Load all components when the app starts
# This dictionary contains everything needed by the UI functions
components = load_global_components()

# Extract individual components from the loaded dictionary
# THESE LINES MUST SUCCESSFULLY ASSIGN VALUES (even if None) ON EVERY RERUN
embeddings = components.get("embeddings")
vectorstore = components.get("vectorstore")
icd_codes = components.get("icd_codes")
icd_embeddings = components.get("icd_embeddings")
doctor_df = components.get("doctor_df")
cases_df = components.get("cases_df")
specialist_categories_list = components.get("specialist_categories_list")
llm = components.get("llm")
followup_chain = components.get("followup_chain")
final_chain = components.get("final_chain")


# --- Helper Functions (adapted from your merged code) ---

# ICD Matching Function (using pre-computed embeddings)
# Requires icd_codes, icd_embeddings, and embeddings_model object from loaded components
def match_icd_codes_dense(query, top_n=3, sim_threshold=0.2, icd_codes=None, icd_embeddings=None, embeddings_model=None):
    """
    Use dense embeddings to match ICD-10 codes semantically.
    Requires loaded ICD embeddings, ICD codes list, and the embedding model object.
    """
    if not query or icd_embeddings is None or embeddings_model is None or not icd_codes:
        # print("Debug: Skipping ICD match - Missing query, embeddings, model, or codes.") # Debug print
        return ""

    try:
        # Use the embedding model object's embed_query method
        query_emb_list = embeddings_model.embed_query(query)
        query_emb = np.array([query_emb_list]).astype('float32')

        # Compute cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(icd_embeddings, query_emb).flatten()

        sorted_indices = np.argsort(sims)[::-1]

        matched = []
        for idx in sorted_indices[:top_n]:
            if sims[idx] > sim_threshold:
                if idx < len(icd_codes):
                     matched.append(icd_codes[idx])
                # else:
                     # print(f"Warning: Invalid index {idx} from ICD similarity search.") # Debug print

        return "; ".join(matched)

    except Exception as e:
        print(f"Error matching ICD codes: {e}")
        traceback.print_exc() # Print detailed traceback
        return "" # Return empty string on error


# RAG Context Retrieval Function
# Requires vectorstore object from loaded components
def retrieve_context(query, vectorstore=None, k=5):
    """
    Retrieves relevant document chunks from the FAISS vectorstore.
    """
    if vectorstore is None:
        print("Vectorstore not loaded, cannot retrieve context.")
        return "RAG context not available.", []

    try:
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
        retrieved_docs = retriever.invoke(query) # Use invoke() for newer LangChain
        context = "\n---\n".join([doc.page_content for doc in retrieved_docs]) # Use separator between docs
        return context.strip(), retrieved_docs

    except Exception as e:
        print(f"Error during RAG context retrieval: {e}")
        traceback.print_exc() # Print detailed traceback
        return f"Error retrieving relevant medical information: {e}", []

# Specialist Extraction Function
# Requires specialist_categories_list from loaded components
def extract_specialist(recommendation_text, specialists_list):
    """
    Extract the specialist category from the recommendation text by matching against
    a provided list of known specialists (case-insensitive match).
    """
    if not recommendation_text or not specialists_list:
        # print("Debug: Skipping specialist extraction - Missing text or list.") # Debug print
        return None # Cannot extract without text or list

    recommendation_lower = recommendation_text.lower()
    specialists_list_sorted = sorted(specialists_list, key=len, reverse=True)

    for specialist in specialists_list_sorted:
        if re.search(r'\b' + re.escape(specialist.lower()) + r'\b', recommendation_lower):
             return specialist # Return the original case specialist name from the list

    # print("Debug: No specific specialist match found in the list.") # Debug print
    return None # No specific specialist found in the list


# Doctor Recommendation Function (with added debug prints)
# Requires doctor_df, cases_df, and embeddings_model from loaded components
def recommend_doctors(category, symptoms, doctor_df=None, cases_df=None, embeddings_model=None):
    """
    Find and recommend top 5 doctors based on the specialist category and patient symptoms.
    Uses loaded doctor/case data and the global embedding model (HuggingFaceEmbeddings instance).
    """
    import faiss

    if doctor_df is None or cases_df is None or embeddings_model is None or not category or not symptoms:
        print("Debug: Skipping doctor recommendation - Missing data, model, category, or symptoms.") # Debug print
        return None # Return None if data/model missing or inputs invalid

    try:
        # Filter for recommended category (case-insensitive match)
        filtered_doctor_df = doctor_df[doctor_df["Specialty"].str.lower() == category.lower()].copy()
        filtered_cases_df = cases_df[cases_df["Specialty"].str.lower() == category.lower()].copy()

        if filtered_cases_df.empty:
            print(f"Debug: No historical cases found for category: {category}")
            return pd.DataFrame() # Return empty DataFrame if no cases for this specialty

        # Ensure necessary columns exist and handle potential NaNs
        required_case_cols = ["Symptom Description", "Doctor ID", "Patient Feedback Rating"]
        if not all(col in filtered_cases_df.columns for col in required_case_cols):
             print("Debug: Required columns missing in patient cases data.")
             return pd.DataFrame()

        filtered_cases_df.dropna(subset=required_case_cols, inplace=True)
        if filtered_cases_df.empty:
             print(f"Debug: No cases with complete data found for category: {category}")
             return pd.DataFrame()


        symptom_texts = filtered_cases_df["Symptom Description"].tolist()
        print(f"Debug: Generating embeddings for {len(symptom_texts)} patient cases in {category}...")

        # Use the embeddings_model object's embed_documents method
        embeddings_list = embeddings_model.embed_documents(symptom_texts)
        embeddings_np = np.array(embeddings_list).astype("float32")
        print(f"Debug: Generated case embeddings shape: {embeddings_np.shape}")

        # --- Create a temporary FAISS index for case similarity search ---
        d = embeddings_np.shape[1]
        temp_index = faiss.IndexFlatL2(d) # Using L2 distance
        temp_index.add(embeddings_np)
        print(f"Debug: Temporary FAISS index built with {temp_index.ntotal} vectors.")

        # Map index to original data index or store core metadata
        case_details_map = []
        for i in range(len(filtered_cases_df)):
             original_df_index = filtered_cases_df.index[i]
             case_details_map.append({
                 "Doctor ID": filtered_cases_df.loc[original_df_index, "Doctor ID"],
                 "Symptom": filtered_cases_df.loc[original_df_index, "Symptom Description"],
                 "Rating": filtered_cases_df.loc[original_df_index, "Patient Feedback Rating"],
             })


        # --- Embed user input symptoms ---
        print(f"Debug: Embedding user symptoms for doctor search: '{symptoms}'")
        query_embedding_list = embeddings_model.embed_query(symptoms)
        query_embedding = np.array([query_embedding_list]).astype("float32")
        print(f"Debug: User symptom embedding shape: {query_embedding.shape}")


        # --- Search top K similar cases ---
        top_k_cases = 10 # Search more cases initially
        if temp_index.ntotal == 0:
             print("Debug: Temporary index is empty.")
             return pd.DataFrame()
        k_to_search = min(top_k_cases, temp_index.ntotal)
        if k_to_search == 0:
             print("Debug: No items to search in temporary index.")
             return pd.DataFrame()

        print(f"Debug: Starting FAISS search for top {k_to_search} cases...")
        distances, indices = temp_index.search(query_embedding, k_to_search)
        print(f"Debug: FAISS search completed. Indices shape: {indices.shape}, Distances shape: {distances.shape}")


        # --- Gather similar cases details ---
        similar_cases_data = []
        print("Debug: Processing FAISS search results...")
        if indices.size > 0 and indices[0][0] != -1:
             for i_idx in indices[0]:
                 if i_idx < 0 or i_idx >= len(case_details_map):
                      print(f"Warning: Received invalid index {i_idx} from FAISS search.")
                      continue
                 case_info = case_details_map[i_idx]
                 case_embedding = embeddings_np[i_idx]
                 sim_score = cosine_similarity(query_embedding, [case_embedding])[0][0]
                 similar_cases_data.append({
                     "Doctor ID": case_info["Doctor ID"],
                     "Symptom": case_info["Symptom"],
                     "Rating": case_info["Rating"],
                     "Similarity Score": round(float(sim_score), 4)
                 })
        else:
             print("Debug: No similar cases found by FAISS search with valid indices.")

        print(f"Debug: Found {len(similar_cases_data)} potential similar cases after initial processing.")

        if not similar_cases_data:
             print(f"Debug: No similar cases data collected for category {category}.")
             return pd.DataFrame()

        similar_cases_df_result = pd.DataFrame(similar_cases_data)
        print(f"Debug: Created similar_cases_df_result with shape {similar_cases_df_result.shape}")


        # --- Aggregate ratings and scores per doctor ---
        similar_cases_df_result.dropna(subset=["Doctor ID", "Rating", "Similarity Score"], inplace=True)
        print(f"Debug: Shape after dropping NaNs for aggregation: {similar_cases_df_result.shape}")

        if similar_cases_df_result.empty:
             print("Debug: No valid similar cases left after dropping NaNs for aggregation.")
             return pd.DataFrame()

        print("Debug: Aggregating scores per doctor...")
        doctor_scores_from_similar = similar_cases_df_result.groupby("Doctor ID").agg(
            avg_rating_from_similar=('Rating', 'mean'),
            num_similar_cases=('Doctor ID', 'count'),
            max_similarity_score=('Similarity Score', 'max')
        ).reset_index()
        print(f"Debug: Aggregated doctor scores shape: {doctor_scores_from_similar.shape}")


        # --- Merge with the main doctor profiles ---
        if "Doctor ID" not in filtered_doctor_df.columns:
            print("Error: 'Doctor ID' column not found in doctor profiles DataFrame.")
            return pd.DataFrame()

        doctor_scores_from_similar['Doctor ID'] = doctor_scores_from_similar['Doctor ID'].astype(str)
        filtered_doctor_df['Doctor ID'] = filtered_doctor_df['Doctor ID'].astype(str)

        print("Debug: Merging aggregated scores with doctor profiles...")
        recommended_doctors = pd.merge(
            doctor_scores_from_similar,
            filtered_doctor_df,
            on="Doctor ID",
            how="left"
        )
        print(f"Debug: Merged recommendations shape: {recommended_doctors.shape}")


        # --- Rank doctors ---
        print("Debug: Ranking doctors...")
        sort_cols = ["max_similarity_score", "avg_rating_from_similar", "num_similar_cases"]
        sort_cols_present = [col for col in sort_cols if col in recommended_doctors.columns]
        if sort_cols_present:
             recommended_doctors = recommended_doctors.sort_values(
                 by=sort_cols_present,
                 ascending=[False] * len(sort_cols_present)
             )


        # --- Select top N doctors ---
        top_n_doctors = 5 # Define variable BEFORE using it
        print(f"Debug: Selecting top {top_n_doctors} doctors...")
        final_recommendation_df = recommended_doctors.head(top_n_doctors).copy()
        print(f"Debug: Final recommendation DF shape: {final_recommendation_df.shape}")


        # --- Clean up column names ---
        print("Debug: Cleaning up column names...")
        final_recommendation_df.rename(columns={
            'avg_rating_from_similar': 'Avg Rating (Similar Cases)',
            'num_similar_cases': 'Similar Cases Found',
            'max_similarity_score': 'Max Similarity Score'
        }, inplace=True)


        # --- Ensure essential display columns are present and handle data types ---
        print("Debug: Ensuring essential display columns and types...")
        display_cols = ["Doctor ID", "Name", "Specialty", "Avg Rating (Similar Cases)",
                        "Max Similarity Score", "Similar Cases Found", "Years of Experience", "Affiliation"]
        for col in display_cols:
            if col not in final_recommendation_df.columns:
                final_recommendation_df[col] = None # Add missing columns with None

        # Convert types carefully
        for col in ['Doctor ID', 'Name', 'Specialty', 'Years of Experience', 'Affiliation']:
            if col in final_recommendation_df.columns:
                 try:
                     final_recommendation_df[col] = final_recommendation_df[col].apply(lambda x: str(x) if pd.notna(x) else "")
                 except Exception as e:
                     print(f"Warning: Could not convert column {col} to string: {e}")
                     final_recommendation_df[col] = "" # Fallback

        for col in ['Avg Rating (Similar Cases)', 'Max Similarity Score']:
             if col in final_recommendation_df.columns:
                  try:
                      final_recommendation_df[col] = pd.to_numeric(final_recommendation_df[col], errors='coerce').round(2)
                      final_recommendation_df[col] = final_recommendation_df[col].fillna(0.0)
                  except Exception as e:
                       print(f"Warning: Could not convert column {col} to numeric: {e}")
                       final_recommendation_df[col] = 0.0 # Fallback

        # Ensure 'Years of Experience' is treated as potentially non-numeric before int conversion
        if 'Years of Experience' in final_recommendation_df.columns:
             try:
                 # Convert to numeric first, coercing errors, fill NaNs, then convert to int
                 final_recommendation_df['Years of Experience'] = pd.to_numeric(final_recommendation_df['Years of Experience'], errors='coerce').fillna(0).astype(int)
             except Exception as e:
                 print(f"Warning: Could not convert 'Years of Experience' to int: {e}")
                 final_recommendation_df['Years of Experience'] = 0 # Fallback

        if 'Similar Cases Found' in final_recommendation_df.columns:
             try:
                 final_recommendation_df['Similar Cases Found'] = pd.to_numeric(final_recommendation_df['Similar Cases Found'], errors='coerce').fillna(0).astype(int)
             except Exception as e:
                  print(f"Warning: Could not convert column 'Similar Cases Found' to int: {e}")
                  final_recommendation_df['Similar Cases Found'] = 0 # Fallback


        print("Debug: Returning final recommendation DataFrame.")
        return final_recommendation_df


    except Exception as e:
        print(f"An error occurred during doctor recommendation: {e}")
        print("--- TRACEBACK ---")
        traceback.print_exc() # Print detailed traceback to console
        print("--- END TRACEBACK ---")
        return None


# --- Streamlit UI ---

#st.title("🩺 MedMatch")
st.markdown('<h1 style="color: #301934;">🩺 MedMatch</h1>', unsafe_allow_html=True)
# st.markdown("##### Fast-Tracking Specialist Access for Smarter, Quicker Care")
st.markdown('<h5 style="color: #9370DB;">Fast-Tracking Specialist Access for Smarter, Quicker Care</h5>', unsafe_allow_html=True)

st.warning("Disclaimer: This is a prototype using AI and data from public/synthetic sources. It is not a substitute for professional medical advice. Always consult with a qualified healthcare professional for any health concerns.")

# --- Initialize Session State ---
# Conversation history for display
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Message history for Ollama API call (includes system prompt and conversation turns)
if 'ollama_messages' not in st.session_state:
    # Define the initial system prompt
    system_prompt = """
    You are a helpful and compassionate AI medical assistant chatbot designed to guide patients based on their symptoms.
    Your primary goal is to help patients determine the type of medical specialist they should consult.

    Engage in a conversation by asking simple, clear follow-up questions about the patient's symptoms.
    Gather details across multiple turns (up to {turn_limit} user turns).
    Aim to understand the nature, location, duration, and severity of symptoms, relevant history, etc.
    If symptoms indicate a medical emergency (e.g., severe chest pain, difficulty breathing), immediately advise seeking emergency medical attention.

    After {turn_limit} user turns, you will be provided with all accumulated symptom details, relevant medical context from a knowledge base, and related medical codes.
    At that point, based *only* on the patient's symptoms and the provided reference material, you will recommend the most appropriate medical specialist category and a brief explanation (1-2 sentences).
    If the information is insufficient for a specific recommendation, state that you cannot make a definitive recommendation and advise seeing a general physician (Primary Care Provider) first.
    Do not include contact information or specific doctor names.

    Maintain a friendly, empathetic, professional, and encouraging tone throughout.
    Do not provide diagnoses, specific treatment advice, or recommend specific doctors during the conversational turns.
    Only provide the specialist recommendation after the required follow-up turns when prompted to do so with the RAG context.

    Let's start. Greet the user and ask them to describe their symptoms.
    """.replace("{turn_limit}", str(CONVERSATION_TURN_LIMIT))

    st.session_state.ollama_messages = [{"role": "system", "content": system_prompt}]
    # Add an initial greeting message from the AI
    initial_greeting = "Hello! Please tell me about your symptoms so I can help you figure out what type of doctor you might need."
    st.session_state.messages.append({"role": "assistant", "content": initial_greeting})
    st.session_state.ollama_messages.append({"role": "assistant", "content": initial_greeting})


# Accumulated symptom details across the conversation turns (for RAG query)
if 'symptom_history' not in st.session_state:
    st.session_state.symptom_history = ""

# Counter for user conversation turns
if 'user_turn_count' not in st.session_state:
    st.session_state.user_turn_count = 0

# Flag to indicate if the final recommendation has been made
if 'recommendation_made' not in st.session_state:
    st.session_state.recommendation_made = False

# Store the final recommended specialist category name
if 'recommended_specialist' not in st.session_state:
    st.session_state.recommended_specialist = None

# Store the final doctor recommendation DataFrame
if 'final_recommendation_df' not in st.session_state:
    st.session_state.final_recommendation_df = None

# Store a flag if no doctors were found
if 'no_doctors_found' not in st.session_state:
    st.session_state.no_doctors_found = None

# --- ADDED: Store context for optional display ---
if 'rag_context_to_show' not in st.session_state:
    st.session_state.rag_context_to_show = None
if 'icd_codes_to_show' not in st.session_state:
    st.session_state.icd_codes_to_show = None
# Optional: Store raw docs if needed later
# if 'retrieved_docs_to_show' not in st.session_state:
#     st.session_state.retrieved_docs_to_show = []
# --- End Add ---

# --- Display Chat Messages ---
# Use a chat container for better scrolling
chat_container = st.container(height=400, border=True) # Added height for scroll
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- Chat Input ---
# Disable input after recommendation is made
input_disabled = st.session_state.recommendation_made
prompt = st.chat_input("Describe your symptoms...", disabled=input_disabled)

# --- Handle User Input ---
if prompt:
    # Add user message to display history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Add user message to Ollama history for conversational context
    st.session_state.ollama_messages.append({"role": "user", "content": prompt})
    # Append user input to symptom history for final RAG query
    st.session_state.symptom_history += f" User: {prompt}\n"
    st.session_state.user_turn_count += 1

    # Display user message in chat
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)

    # --- Clear previous recommendation results before generating new ones ---
    if 'final_recommendation_df' in st.session_state:
        del st.session_state.final_recommendation_df
    if 'no_doctors_found' in st.session_state:
        del st.session_state.no_doctors_found
    # --- End Clear ---

    # Rerun the script to clear input and show updated chat & trigger AI response
    st.rerun()


# --- AI Response Logic (Runs on each rerun, controlled by state) ---
# Only proceed if the last message is from the user and we haven't made a recommendation yet
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user" and not st.session_state.recommendation_made:

     # Check if essential LLM components are loaded
     if llm is None or followup_chain is None or final_chain is None:
         with chat_container:
              with st.chat_message("assistant"):
                   error_msg = "AI conversation service is not available due to missing components."
                   st.error(error_msg)
                   # Add this message to history to persist
                   st.session_state.messages.append({"role": "assistant", "content": error_msg})
              # Do NOT rerun or continue, wait for user/fix

     # Determine if it's time for the final recommendation or another follow-up
     elif st.session_state.user_turn_count >= CONVERSATION_TURN_LIMIT:
         # --- Time for RAG and Final Recommendation ---
         with chat_container:
             with st.chat_message("assistant"):
                 with st.spinner("Analyzing details and finding the best specialist..."):
                     try:
                         # Ensure essential RAG/ICD components are loaded
                         if vectorstore is None:
                             rag_context = "RAG retrieval service is not available."
                             retrieved_docs = []
                             print(rag_context); st.warning(rag_context) # Display warning in chat
                         else:
                             rag_context, retrieved_docs = retrieve_context(
                                 st.session_state.symptom_history.strip(), vectorstore=vectorstore, k=RAG_K)
                             print("Debug: Retrieved RAG context.")

                         if icd_embeddings is not None and embeddings is not None and icd_codes is not None:
                             query_icd_codes_str = match_icd_codes_dense(
                                 st.session_state.symptom_history.strip(), top_n=5, sim_threshold=0.2,
                                 icd_codes=icd_codes, icd_embeddings=icd_embeddings, embeddings_model=embeddings)
                             if not query_icd_codes_str: query_icd_codes_str = "None found."
                         else:
                             query_icd_codes_str = "ICD mapping not available."
                             print(query_icd_codes_str); st.warning(query_icd_codes_str) # Display warning in chat

                         # --- Store context for optional display ---
                         st.session_state.rag_context_to_show = rag_context
                         st.session_state.icd_codes_to_show = query_icd_codes_str
                         # Optional: store retrieved docs if needed
                         # st.session_state.retrieved_docs_to_show = retrieved_docs
                         print("DEBUG UI: Stored RAG context and ICD codes in session state.")
                         # --- End Store context ---


                         # Use the final LangChain chain to get the recommendation from LLM
                         ai_response_obj = final_chain.invoke(
                             input={"symptoms": st.session_state.symptom_history.strip(), "context": rag_context, "icd_codes": query_icd_codes_str}
                         )
                         ai_response = ai_response_obj.get('text', 'Error: No text response from LLM.').strip()
                         print("Debug: LLM recommendation generated.")

                         # Display the AI response (Specialist Recommendation) INSIDE the chat bubble
                         st.markdown("---")
                         st.markdown("**Specialist Recommendation:**")
                         st.markdown(ai_response)

                         # --- GENERATE Doctor Recommendation Data (but don't display here) ---
                         # Check if essential components are loaded before proceeding
                         if doctor_df is not None and cases_df is not None and embeddings is not None and specialist_categories_list:
                             extracted_specialist = extract_specialist(ai_response, specialist_categories_list)
                             if extracted_specialist:
                                 print(f"Debug: Extracted specialist category: {extracted_specialist}")
                                 st.session_state.recommended_specialist = extracted_specialist # Store name

                                 # Call function to get the DataFrame
                                 recommended_doctors_df = recommend_doctors(
                                     extracted_specialist, st.session_state.symptom_history.strip(),
                                     doctor_df=doctor_df, cases_df=cases_df, embeddings_model=embeddings)

                                 # --- STORE result in session state ---
                                 if recommended_doctors_df is not None and not recommended_doctors_df.empty:
                                     st.session_state.final_recommendation_df = recommended_doctors_df
                                     print("DEBUG UI: Stored recommendation DF in session state.")
                                 else:
                                     # Store the specialist name for whom no doctors were found
                                     st.session_state.no_doctors_found = extracted_specialist
                                     print(f"DEBUG UI: No doctors found for {extracted_specialist}, setting flag.")
                             else:
                                 print(f"Debug: Could not extract specialist from: {ai_response}")
                                 st.info("Could not pinpoint a specific specialist category for doctor search.") # Display info in chat
                                 st.session_state.no_doctors_found = "Unknown" # Set flag for unknown
                         else:
                             print("Debug: Doctor data, embeddings, or specialist list missing. Skipping doctor generation.")
                             st.warning("Doctor recommendation data missing. Skipping doctor search.") # Display warning in chat

                         # NOTE: Removed the expander from here as it will be displayed separately later

                     except Exception as e:
                         print(f"Error during final LLM recommendation chain run: {e}")
                         error_msg = f"An error occurred while generating the recommendation: {e}"
                         st.error(error_msg) # Display error in chat bubble
                         traceback.print_exc() # Print traceback to console


                 # --- Update State & Add Final Message (ONLY specialist text) ---
                 st.session_state.recommendation_made = True
                 # Add the specialist recommendation text to display history
                 # Ensure ai_response is defined even if an error occurred earlier in the try block
                 if 'ai_response' not in locals(): ai_response = "Error in recommendation." # Fallback
                 final_message_content = f"**Specialist Recommendation:**\n{ai_response}" # Reuse the variable
                 st.session_state.messages.append({"role": "assistant", "content": final_message_content})

                 # --- RERUN to update state and proceed to the SEPARATE display section ---
                 st.rerun()


     elif st.session_state.messages[-1]["role"] == "user": # Only generate AI response if the last message is from the user and not in final stage
         # --- Continue Conversation with Ollama (Follow-up) ---
         with chat_container:
             with st.chat_message("assistant"):
                 with st.spinner("Thinking..."):
                     try:
                          # Use the follow-up chain
                          response_obj = followup_chain.invoke(
                              input={"symptoms": st.session_state.symptom_history.strip()}
                          )
                          ai_response = response_obj.get('text', 'Error: No text response from LLM.').strip()
                          print("Debug: Follow-up question generated.")
                          st.markdown(ai_response) # Display follow-up question
                          # Add AI response to both display and Ollama history
                          st.session_state.messages.append({"role": "assistant", "content": ai_response})
                          st.session_state.ollama_messages.append({"role": "assistant", "content": ai_response})

                     except Exception as e:
                          print(f"Error during follow-up chain run: {e}")
                          error_msg = f"An error occurred while generating the next question: {e}"
                          st.error(error_msg) # Display error in UI
                          traceback.print_exc()

                 # IMPORTANT: After generating AI response, rerun to update display and state
                 st.rerun()

# --- SEPARATE DISPLAY SECTION for Doctor Recommendations & Context ---
# This runs AFTER the main chat logic and potential reruns

# Check if recommendation is made FIRST
if st.session_state.recommendation_made:

    # --- Display Context Expander ---
    with st.expander("🔍 Show Analysis Details (Context & Codes)"):
        # Retrieve stored context from session state
        rag_context_display = st.session_state.get('rag_context_to_show', "Context not generated or available.")
        icd_codes_display = st.session_state.get('icd_codes_to_show', "ICD codes not generated or available.")

        st.markdown("**Symptoms Used for Analysis:**")
        st.text(st.session_state.get('symptom_history', "Symptoms not available.").strip())

        st.markdown("**Matched ICD10 Codes:**")
        st.text(icd_codes_display)

        st.markdown("**Retrieved Context from Knowledge Base (Top Chunks):**")
        # Check if rag_context_display is a valid, non-error string
        if isinstance(rag_context_display, str) and rag_context_display.strip() and rag_context_display != "RAG context not available." and not rag_context_display.startswith("Error retrieving"):
             # Split context back into chunks using the separator used in retrieve_context
             chunks = rag_context_display.split("\n---\n")
             # Display first RAG_K chunks (adjust if you retrieve more initially)
             for i, chunk_text in enumerate(chunks[:RAG_K]): # Limit display if needed
                 st.markdown(f"--- Chunk {i+1} ---")
                 # Use text_area for better formatting and potential scrolling of long chunks
                 st.text_area(f"Chunk {i+1}", value=chunk_text.strip(), height=150, disabled=True, key=f"rag_chunk_{i}")
        elif isinstance(rag_context_display, str):
             st.text(rag_context_display) # Show error message if RAG failed or context is unavailable
        else:
             st.text("No relevant context found or generated.") # Handle None case
    # --- End Context Expander ---


    # --- Display Doctor Recommendations (if available) ---
    if 'final_recommendation_df' in st.session_state:
        recommended_doctors_df_display = st.session_state.final_recommendation_df # Get DF from state
        if recommended_doctors_df_display is not None and not recommended_doctors_df_display.empty:
            st.markdown("---") # Separator
            st.markdown("\n**✅ Top Recommended Doctors (based on similar cases):**")
            display_cols = [
                "Doctor ID", "Name", "Specialty", "Avg Rating (Similar Cases)",
                "Max Similarity Score", "Similar Cases Found", "Years of Experience", "Affiliation"
            ]
            display_cols_present = [col for col in display_cols if col in recommended_doctors_df_display.columns]

            print("DEBUG DISPLAY: DataFrame columns:", recommended_doctors_df_display.columns.tolist())
            print("DEBUG DISPLAY: Selected columns:", display_cols_present)

            if not display_cols_present:
                st.warning("Warning: No matching columns found in the generated doctor data to display.")
                print("DEBUG DISPLAY: display_cols_present is empty!")
            else:
                try:
                    st.dataframe(recommended_doctors_df_display[display_cols_present], hide_index=True)
                except Exception as display_error:
                    print(f"DEBUG DISPLAY: Error during st.dataframe: {display_error}")
                    st.error(f"An error occurred while displaying the doctor recommendations table: {display_error}")
                    traceback.print_exc()
                    st.write("Raw recommendation data (fallback):", recommended_doctors_df_display) # Fallback

    # --- Display "No Doctors Found" Message (if applicable) ---
    elif 'no_doctors_found' in st.session_state:
        specialist_name = st.session_state.no_doctors_found
        if specialist_name and specialist_name != "Unknown": # Check flag is set and not 'Unknown'
            st.info(f"No doctor profiles found for the **{specialist_name}** specialty that match your symptoms in the historical data.")
        elif specialist_name == "Unknown":
             st.info("Could not determine a specific specialist category to search for matching doctors.")


    # --- Display Canned Response (always last if recommendation made) ---
    canned_response = "Our recommendation has been provided above. Please consult a healthcare professional for further guidance."
    # Check if messages list is not empty before accessing the last element
    last_message_content = st.session_state.messages[-1].get("content", "") if st.session_state.messages else ""

    if last_message_content != canned_response:
        # Display it within the chat container for consistency
        with chat_container:
             with st.chat_message("assistant"):
                 st.markdown(canned_response)
                 # Add to messages history so it persists
                 st.session_state.messages.append({"role": "assistant", "content": canned_response})

# --- Sidebar ---
st.sidebar.header("About")
st.sidebar.info(
    "This is a prototype AI Healthcare Assistant demonstrating symptom-based specialist triage "
    "using RAG with Ollama and doctor recommendation based on historical case similarity."
)

st.sidebar.header("Data Sources")
st.sidebar.markdown(f"- PDFs from '{DATA_FOLDER}'")
st.sidebar.markdown(f"- ICD-10 Mapping from '{ICD_CSV_PATH}'")
st.sidebar.markdown(f"- Synthetic/Sample Doctor/Case data from '{DOCTOR_PROFILES_PATH}' and '{PATIENT_CASES_PATH}'")
st.sidebar.markdown("*(Note: Data sources are for demonstration and may not be comprehensive or real patient data)*")

# --- Restart Button ---
if st.sidebar.button("Restart Chat"):
    # Reset all relevant session state variables
    st.session_state.messages = []
    system_prompt_content = ""
    # Check if ollama_messages exists and has at least one element
    if 'ollama_messages' in st.session_state and st.session_state.ollama_messages:
        if st.session_state.ollama_messages[0]['role'] == 'system':
            system_prompt_content = st.session_state.ollama_messages[0]['content']
    st.session_state.ollama_messages = [{"role": "system", "content": system_prompt_content}] if system_prompt_content else []

    st.session_state.symptom_history = ""
    st.session_state.user_turn_count = 0
    st.session_state.recommendation_made = False
    st.session_state.recommended_specialist = None

    # --- Crucially, clear the stored dataframe, flags, AND CONTEXT on restart ---
    if 'final_recommendation_df' in st.session_state:
        del st.session_state.final_recommendation_df
    if 'no_doctors_found' in st.session_state:
        del st.session_state.no_doctors_found
    # --- Add clearing for context ---
    if 'rag_context_to_show' in st.session_state:
        del st.session_state.rag_context_to_show
    if 'icd_codes_to_show' in st.session_state:
        del st.session_state.icd_codes_to_show
    # if 'retrieved_docs_to_show' in st.session_state: del st.session_state.retrieved_docs_to_show # If using this
    # --- End clear context ---

    # Add back the initial greeting ONLY if system prompt was successfully retrieved
    if st.session_state.ollama_messages:
        initial_greeting = "Hello! Please tell me about your symptoms so I can help you figure out what type of doctor you might need."
        st.session_state.messages.append({"role": "assistant", "content": initial_greeting})
        st.session_state.ollama_messages.append({"role": "assistant", "content": initial_greeting})

    # Rerun the app to apply the reset state and show the initial greeting
    st.rerun()