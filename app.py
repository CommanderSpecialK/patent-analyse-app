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
    if "GEMINI_API_KEY" not in st.secrets:
        st.error("❌ Der GEMINI_API_KEY fehlt in den Streamlit Secrets!")
        return None
    api_key = st.secrets["GEMINI_API_KEY"].strip().strip('"').strip("'")
    return genai.Client(api_key=api_key)

# --- GEMINI EMBEDDING BERECHNUNG (Strikt mathematisch gedrosselt) ---
def get_gemini_embeddings(texts, model_name="gemini-embedding-001"):
    """Erzeugt hochpräzise Vektoren via Gemini API unter strikter Einhaltung des 1000-Texte-Limits."""
    if not texts:
        return np.array([])
        
    client = get_gemini_client()
    if client is None:
        return np.array([])
        
    embeddings = []
    # Auf 10 reduziert, um die Last perfekt über die Zeit zu strecken
    batch_size = 10 
    
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
                
                # Feste Pause von 1,5 Sekunden nach 10 Texten = Konstant max. 400 Texte/Minute.
                # Das reißt das Google-Limit (1.000) mathematisch garantiert niemals!
                time.sleep(1.5)
                break  
                
            except errors.APIError as e:
                if e.code == 429:
                    if versuch < 4:
                        countdown_placeholder = st.empty()
                        # Falls durch vorherige Klicks noch Rest-Last auf der Leitung liegt, leeren wir sie mit 65s
                        for sekunde in range(65, -1, -1):
                            countdown_placeholder.warning(
                                f"⏳ **Google API-Limit erreicht.** Die App regeneriert die Verbindung. "
                                f"Weiter in **{sekunde} Sekunden**... (Versuch {versuch+1}/5)"
                            )
                            time.sleep(1)
                        countdown_placeholder.empty()
                        continue
                st.error(f"⚠️ Kritischer API-Fehler (Code {e.code}): {e.message}")
                return np.array([])
            except Exception as e:
                st.error(f"⚠️ Unerwarteter Fehler: {e}")
                return np.array([])
            
    return np.array(embeddings)


# --- OPENALEX API HILFSFUNKTION ---
def search_openalex_patents(query_string, filter_criterion, score_threshold, max_results=100):
    clean_query = query_string.replace(" AND ", " ").replace(" OR ", " ").replace("'", "")
    url = f"https://openalex.org{clean_query}&filter=type:patent&per_page={max_results}"
    patents_found = []
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            if not map: return []
                
            criterion_embeddings = get_gemini_embeddings([filter_criterion])
            if criterion_embeddings.size == 0: return []
            criterion_embedding = criterion_embeddings
            
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
                                for pos in positions: abstract_words[pos] = word
                        if abstract_words:
                            abstract = " ".join([abstract_words[p] for p in sorted(abstract_words.keys())])
                    except: pass
                
                display_name = work.get("display_name", "")
                patent_id = display_name.replace("Patent: ", "") if display_name else work.get("id", "").split("/")[-1]
                espacenet_url = f"https://espacenet.com{patent_id.replace('-', '').replace(' ', '')}"
                
                patent_texts.append(f"{title}. {abstract}")
                patent_metadata.append({"id": patent_id, "title": title, "abstract": abstract, "url": espacenet_url})
            
            patent_embeddings = get_gemini_embeddings(patent_texts)
            if patent_embeddings.size == 0: return []
            
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
                return pd.DataFrame(patents_found).sort_values(by="raw_score", ascending=False).drop(columns=["raw_score"]).to_dict('records')
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
        st.title("💡 Patent Analyse Tool (Zweistufiger Vektor-Schutz)")
        st.write("Lade zwei Excel-Listen hoch. Die Schritte sind getrennt, um Google-Limits (429) im kostenlosen Tarif strikt zu vermeiden.")
        
        col1, col2 = st.columns(2)
        with col1: uploaded_file_ext = st.file_uploader("Excel-Liste hochladen (Extern)", type=["xlsx", "xlsm"])
        with col2: uploaded_file_own = st.file_uploader("Excel-Liste hochladen (Eigene)", type=["xlsx", "xlsm"])
        
        def load_patent_data(f):
            if f is not None:
                df = pd.read_excel(f, engine="openpyxl")
                df.columns = ['Patentnummer', 'Titel_Original', 'Titel_Uebersetzt', 'Zusammenfassung_Uebersetzt'] + list(df.columns[4:])
                return df.fillna("")
            return None

        df_ext = load_patent_data(uploaded_file_ext)
        df_own = load_patent_data(uploaded_file_own)

        if df_ext is not None and df_own is not None:
            score_threshold = st.slider("Mindest-Score für Relevanz (in %)", 0, 100, 60, key="slider1")
            
            # Speicher im Session State sauber initialisieren
            if "emb_ext" not in st.session_state: st.session_state["emb_ext"] = None
            if "emb_own" not in st.session_state: st.session_state["emb_own"] = None

            # Statusanzeige im Sidebar-Bereich
            st.sidebar.markdown("### 📈 Analyse-Status")
            status_ext_msg = "❌ Noch nicht berechnet" if st.session_state["emb_ext"] is None else "✅ Fertig im Speicher"
            status_own_msg = "❌ Noch nicht berechnet" if st.session_state["emb_own"] is None else "✅ Fertig im Speicher"
            st.sidebar.write(f"**Externe Liste:** {status_ext_msg}")
            st.sidebar.write(f"**Eigene Liste:** {status_own_msg}")

            if st.sidebar.button("🔄 Gesamten Cache löschen"):
                st.session_state["emb_ext"] = None
                st.session_state["emb_own"] = None
                st.rerun()

            # --- SCHRITT 1: EXTERNE LISTE ---
            if st.session_state["emb_ext"] is None:
                st.info("👉 **Schritt 1:** Klicke zuerst hier, um die externe Liste einzulesen.")
                if st.button("▶️ Schritt 1: Externe Liste verarbeiten"):
                    with st.spinner("Hole Gemini-Vektoren für die externe Liste..."):
                        texts_ext = (df_ext['Titel_Uebersetzt'].astype(str) + " " + df_ext['Zusammenfassung_Uebersetzt'].astype(str)).tolist()
                        temp_emb = get_gemini_embeddings(texts_ext)
                        
                        # Absicherung: Nur speichern, wenn die Berechnung fehlerfrei geklappt hat!
                        if temp_emb.size > 0:
                            st.session_state["emb_ext"] = temp_emb
                            st.rerun()
                        else:
                            st.error("❌ Schritt 1 abgebrochen wegen eines API-Fehlers. Bitte kurz warten und erneut versuchen.")


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

