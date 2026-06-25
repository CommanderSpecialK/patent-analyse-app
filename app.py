import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from google import genai
from google.genai import errors

# Seiteneinstellungen
st.set_page_config(page_title="Patent Analyse Tool", layout="wide")

# --- PASSTWORT SCHUTZ FUNKTION ---
def check_password():
    if "password_correct" not in st.session_state:
        st.title("🔒 Login erforderlich")
        st.text_input("Bitte gib das Passwort ein:", type="password", on_change=lambda: st.session_state.update({"password_correct": st.session_state["password"] == st.secrets["APP_PASSWORD"]}), key="password")
        return False
    return st.session_state["password_correct"]

# --- GEMINI CLIENT INITIALISIERUNG ---
def get_gemini_client():
    """Holt den Key explizit aus Streamlit Secrets und initialisiert den Client."""
    if "GEMINI_API_KEY" not in st.secrets:
        st.error("❌ Der GEMINI_API_KEY fehlt in den Streamlit Secrets!")
        return None
    
    api_key = st.secrets["GEMINI_API_KEY"].strip().strip('"').strip("'")
    return genai.Client(api_key=api_key)

# --- GEMINI EMBEDDING BERECHNUNG ---
def get_gemini_embeddings(texts, model_name="gemini-embedding-001"):
    """Erzeugt hochpräzise Vektoren via Gemini API unter strikter Einhaltung des 1000-Texte-Limits."""
    if not texts:
        return np.array([])
        
    client = get_gemini_client()
    if client is None:
        return np.array([])
        
    embeddings = []
    batch_size = 100 
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_texts = [str(t).strip() if str(t).strip() != "" else "Kein Text vorhanden" for t in batch_texts]
        
        for versuch in range(5):
            try:
                response = client.models.embed_content(
                    model=model_name,
                    contents=batch_texts
                )
                for embedding in response.embeddings:
                    embeddings.append(embedding.values)
                
                # Feste Pause zwischen den 100er-Blöcken
                time.sleep(6.2)
                break  
                
            except errors.APIError as e:
                if e.code == 429:
                    if versuch < 4:
                        countdown_placeholder = st.empty()
                        
                        # Da das Limit nur minimal überschritten wurde, reichen oft kurze Pausen.
                        # Wir warten hier sicherheitshalber 20 Sekunden, um die Quote zu leeren.
                        for sekunde in range(20, -1, -1):
                            countdown_placeholder.warning(
                                f"⏳ **Google API-Limit kurzzeitig erreicht.** Die App pausiert kurz zur Entlastung. "
                                f"Weiter geht es automatisch in **{sekunde} Sekunden**... (Versuch {versuch+1}/5)"
                            )
                            time.sleep(1)
                        countdown_placeholder.empty()
                        continue
                
                st.error(f"⚠️ Kritischer API-Fehler (Code {e.code}): {e.message}")
                return np.array([])
                
            except Exception as e:
                st.error(f"⚠️ Unerwarteter Fehler bei der Gemini-API-Abfrage: {e}")
                return np.array([])
            
    return np.array(embeddings)



# --- OPENALEX API HILFSFUNKTION ---
def search_openalex_patents(query_string, filter_criterion, score_threshold, max_results=100):
    """Sucht Patente via OpenAlex und vergleicht sie mittels Gemini Embeddings."""
    clean_query = query_string.replace(" AND ", " ").replace(" OR ", " ").replace("'", "")
    url = f"https://openalex.org{clean_query}&filter=type:patent&per_page={max_results}"
    patents_found = []
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            if not results: 
                return []
                
            # Gemini Vektor für das Filter-Kriterium berechnen
            criterion_embeddings = get_gemini_embeddings([filter_criterion])
            if criterion_embeddings.size == 0:
                return []
            criterion_embedding = criterion_embeddings[0]
            
            patent_texts = []
            patent_metadata = []
            
            for work in results:
                title = work.get("title", "Kein Titel verfügbar")
                abstract = "Keine Zusammenfassung verfügbar"
                abstract_inverted = work.get("abstract_inverted")
                if abstract_inverted and isinstance(abstract_inverted, dict):
                    try:
                        abstract_words = {}
                        for word, positions in abstract_inverted.items():
                            if positions and isinstance(positions, list):
                                for pos in positions: 
                                    abstract_words[pos] = word
                        if abstract_words:
                            abstract = " ".join([abstract_words[p] for p in sorted(abstract_words.keys())])
                    except: 
                        pass
                
                display_name = work.get("display_name", "")
                patent_id = display_name.replace("Patent: ", "") if display_name else work.get("id", "").split("/")[-1]
                espacenet_url = f"https://espacenet.com{patent_id.replace('-', '').replace(' ', '')}"
                
                patent_texts.append(f"{title}. {abstract}")
                patent_metadata.append({"id": patent_id, "title": title, "abstract": abstract, "url": espacenet_url})
            
            # Alle Patent-Vektoren auf einmal via Gemini holen
            patent_embeddings = get_gemini_embeddings(patent_texts)
            if patent_embeddings.size == 0:
                return []
            
            # Ähnlichkeiten berechnen
            for idx, patent_embedding in enumerate(patent_embeddings):
                sim = np.dot(criterion_embedding, patent_embedding) / (np.linalg.norm(criterion_embedding) * np.linalg.norm(patent_embedding))
                percentage_score = round(sim * 100, 1)
                
                if percentage_score >= score_threshold:
                    meta = patent_metadata[idx]
                    patents_found.append({
                        "Patentnummer": meta["id"],
                        "Titel": meta["title"],
                        "Zusammenfassung (Abstract)": meta["abstract"] if len(meta["abstract"]) < 150 else meta["abstract"][:150] + "...",
                        "KI Relevanz Score": f"{percentage_score} %",
                        "Link zur Quelle": meta["url"],
                        "raw_score": percentage_score
                    })
            
            if patents_found:
                df_temp = pd.DataFrame(patents_found)
                return df_temp.sort_values(by="raw_score", ascending=False).drop(columns=["raw_score"]).to_dict('records')
        return []
    except Exception as e:
        st.error(f"Fehler bei der OpenAlex-Abfrage: {e}")
        return []

# --- APP START ---
if check_password():
    tab_vergleich, tab_suche = st.tabs(["📊 Patent-Listen Vergleich", "🔍 Live-Recherche (OpenAlex Massen-Filter)"])

    # =========================================================================
    # REITER 1: PATENT-LISTEN VERGLEICH
    # =========================================================================
    with tab_vergleich:
        st.title("💡 Patent Analyse Tool (Gemini-Power)")
        st.write("Lade zwei Excel-Listen hoch, um sie mit modernster Gemini-Semantik zu vergleichen.")
        
        col1, col2 = st.columns(2)
        with col1: uploaded_file_ext = st.file_uploader("Excel-Liste hochladen (Extern)", type=["xlsx", "xlsm"])
        with col2: uploaded_file_own = st.file_uploader("Excel-Liste hochladen (Eigene)", type=["xlsx", "xlsm"])
        
        def load_patent_data(f):
            if f is not None:
                df = pd.read_excel(f, engine="openpyxl")
                df.columns = ['Patentnummer', 'Titel_Original', 'Titel_Uebersetzt', 'Zusammenfassung_Uebersetzt'] + list(df.columns[4:])
                return df.fillna("")
        df_ext = load_patent_data(uploaded_file_ext)
        df_own = load_patent_data(uploaded_file_own)

        if df_ext is not None and df_own is not None:
            score_threshold = st.slider("Mindest-Score für Relevanz (in %)", 0, 100, 60, key="slider1") # Höherer Standard-Wert, da Gemini präziser trennt
            if st.button("Semantische Nähe berechnen"):
                with st.spinner("Gemini analysiert die Patente... Bitte warten."):
                    
                    # Texte vorbereiten
                    texts_ext = (df_ext['Titel_Uebersetzt'].astype(str) + " " + df_ext['Zusammenfassung_Uebersetzt'].astype(str)).tolist()
                    texts_own = (df_own['Titel_Uebersetzt'].astype(str) + " " + df_own['Zusammenfassung_Uebersetzt'].astype(str)).tolist()
                    
                    # Embeddings über Gemini API abrufen
                    emb_ext = get_gemini_embeddings(texts_ext)
                    emb_own = get_gemini_embeddings(texts_own)
                    
                    results = []
                    for idx_ext, e_ext in enumerate(emb_ext):
                        best_score = 0
                        best_id, best_title = "", ""
                        for idx_own, e_own in enumerate(emb_own):
                            sim = np.dot(e_ext, e_own) / (np.linalg.norm(e_ext) * np.linalg.norm(e_own))
                            if sim > best_score: 
                                best_score, best_id, best_title = sim, df_own.iloc[idx_own]['Patentnummer'], df_own.iloc[idx_own]['Titel_Uebersetzt']
                        
                        score_percent = round(best_score * 100, 1)
                        if score_percent >= score_threshold:
                            results.append({
                                "Externes Patent": df_ext.iloc[idx_ext]['Patentnummer'], 
                                "Titel (Extern)": df_ext.iloc[idx_ext]['Titel_Uebersetzt'], 
                                "Ähnlichstes eigenes Patent": best_id, 
                                "Titel (Eigen)": best_title, 
                                "Match Score": f"{score_percent} %"
                            })
                            
                    if results: 
                        st.dataframe(pd.DataFrame(results), use_container_width=True)
                    else:
                        st.info("Keine Patente über dem gewählten Mindest-Score gefunden.")


    # =========================================================================
    # REITER 2: LIVE-RECHERCHE (OPENALEX)
    # =========================================================================
    with tab_suche:
        st.title("🔍 Live-Recherche & Massen-Filter")
        st.write("Durchsuche die weltweite OpenAlex-Datenbank nach Patenten und filtere sie live per Gemini-Semantik.")
        
        col1, col2 = st.columns(2)
        with col1:
            query_input = st.text_input("Suchbegriff für OpenAlex (z.B. 'Solid state battery')", "Solid state battery")
            max_res = st.slider("Maximale Treffer von OpenAlex", 10, 200, 50, step=10)
        with col2:
            criterion_input = st.text_input("KI-Filter-Kriterium (Worauf soll geprüft werden?)", "Anode materials made of silicon")
            score_threshold_suche = st.slider("Mindest-Score für Relevanz (in %)", 0, 100, 50, key="slider2")
            
        if st.button("Recherche & KI-Analyse starten"):
            with st.spinner("Frage OpenAlex ab und berechne Gemini-Relevanz..."):
                daten = search_openalex_patents(query_input, criterion_input, score_threshold_suche, max_results=max_res)
                if daten:
                    st.success(f"{len(daten)} relevante Patente gefunden!")
                    st.dataframe(pd.DataFrame(daten), use_container_width=True)
                else:
                    st.info("Keine Patente gefunden, die den Suchbegriffen und dem KI-Filter entsprechen.")

