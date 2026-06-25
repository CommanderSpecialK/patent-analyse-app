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

# --- GEMINI CLIENT INITIALISIERUNG MIT ROTATION ---
def get_gemini_client_by_index(key_index):
    """Initialisiert den Client mit dem Schlüssel des aktuellen Index."""
    secret_name = f"GEMINI_API_KEY_{key_index}"
    if secret_name not in st.secrets:
        return None
    api_key = st.secrets[secret_name].strip().strip('"').strip("'")
    return genai.Client(api_key=api_key)

# --- GEMINI EMBEDDING BERECHNUNG MIT AUTOMATISCHER KEY-ROTATION ---
def get_gemini_embeddings(texts, model_name="gemini-embedding-001"):
    if not texts:
        return np.array([])
        
    embeddings = []
    batch_size = 5  # Bleibt im sicheren Schildkröten-Modus
    
    progress_bar = st.progress(0)
    total_chunks = max(1, int(np.ceil(len(texts) / batch_size)))
    
    # Wir starten standardmäßig mit dem ersten Key (Index 1)
    if "current_key_index" not in st.session_state:
        st.session_state["current_key_index"] = 1
        
    for current_chunk, i in enumerate(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]
        batch_texts = [str(t).strip() if str(t).strip() != "" else "Kein Text vorhanden" for t in batch_texts]
        
        erfolgreich = False
        # Wir versuchen es maximal so oft, wie wir Keys zur Verfügung haben
        for key_versuch in range(3):
            client = get_gemini_client_by_index(st.session_state["current_key_index"])
            
            if client is None:
                # Falls Key 2 oder 3 nicht existiert, rotieren wir zurück zu Key 1
                st.session_state["current_key_index"] = 1
                client = get_gemini_client_by_index(1)
            
            try:
                response = client.models.embed_content(
                    model=model_name,
                    contents=batch_texts
                )
                for embedding in response.embeddings:
                    embeddings.append(embedding.values)
                
                progress_bar.progress((current_chunk + 1) / total_chunks)
                time.sleep(3.0)
                erfolgreich = True
                break  # Paket war erfolgreich, weiter zum nächsten Block
                
            except errors.APIError as e:
                # Falls Limit erreicht (429), rotieren wir SOFORT zum nächsten Key
                if e.code == 429:
                    altes_projekt = st.session_state["current_key_index"]
                    # Wechsel von 1 -> 2, von 2 -> 3, von 3 -> 1
                    st.session_state["current_key_index"] = 2 if altes_projekt == 1 else (3 if altes_projekt == 2 else 1)
                    st.warning(f"🔄 Limit bei Key {altes_projekt} erreicht! Schalte nahtlos um auf Key {st.session_state['current_key_index']}...")
                    time.sleep(1) # Kurze Gedenksekunde für den Wechsel
                    continue  # Wiederholt den exakt selben Text-Block mit dem neuen Key
                else:
                    st.error(f"⚠️ Kritischer API-Fehler (Code {e.code}): {e.message}")
                    return np.array([])
            except Exception as e:
                st.error(f"⚠️ Unerwarteter Fehler: {e}")
                return np.array([])
                
        if not erfolgreich:
            st.error("❌ Alle verfügbaren API-Schlüssel wurden blockiert. Bitte warte kurz.")
            return np.array([])
            
    progress_bar.empty()
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
            if not results: return []
                
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
    # REITER 1: PATENT-LISTEN VERGLEICH (TEIL 2)
    # =========================================================================
    with tab_vergleich:
        st.title("💡 Patent Analyse Tool (Zweistufiger Vektor-Schutz)")
        st.write("Lade zwei Excel-Listen hoch, um sie mit modernster Gemini-Semantik zu vergleichen.")
        
        if "emb_ext" not in st.session_state: st.session_state["emb_ext"] = None
        if "emb_own" not in st.session_state: st.session_state["emb_own"] = None
        
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
            
            st.sidebar.markdown("### 📈 Analyse-Status")
            status_ext_msg = "❌ Offen" if st.session_state["emb_ext"] is None else "✅ Im Speicher"
            status_own_msg = "❌ Offen" if st.session_state["emb_own"] is None else "✅ Im Speicher"
            st.sidebar.write(f"**Externe Liste:** {status_ext_msg}")
            st.sidebar.write(f"**Eigene Liste:** {status_own_msg}")

            if st.sidebar.button("🔄 Gesamten Cache löschen"):
                st.session_state["emb_ext"] = None
                st.session_state["emb_own"] = None
                st.experimental_rerun()

            # --- SCHRITT 1: EXTERNE LISTE ---
            if st.session_state["emb_ext"] is None:
                st.info("👉 **Schritt 1:** Klicke hier, um die externe Liste einzulesen.")
                if st.button("▶️ Schritt 1: Externe Liste verarbeiten", key="btn_schritt1"):
                    with st.spinner("Hole Gemini-Vektoren für die externe Liste..."):
                        texts_ext = (df_ext['Titel_Uebersetzt'].astype(str) + " " + df_ext['Zusammenfassung_Uebersetzt'].astype(str)).tolist()
                        temp_emb = get_gemini_embeddings(texts_ext)
                        if temp_emb.size > 0:
                            st.session_state["emb_ext"] = temp_emb
                            st.success("🎉 Externe Liste erfolgreich verarbeitet!")
                            st.rerun()
                        else:
                            st.error("❌ Fehler bei der Berechnung von Liste 1.")
            # --- SCHRITT 2: EIGENE LISTE ---
            if st.session_state["emb_ext"] is not None and st.session_state["emb_own"] is None:
                st.info("👉 **Schritt 2:** Externe Liste ist bereit. Klicke hier für deine eigene Liste.")
                if st.button("▶️ Schritt 2: Eigene Liste verarbeiten", key="btn_schritt2"):
                    with st.spinner("Hole Gemini-Vektoren für die eigene Liste..."):
                        texts_own = (df_own['Titel_Uebersetzt'].astype(str) + " " + df_own['Zusammenfassung_Uebersetzt'].astype(str)).tolist()
                        temp_emb_own = get_gemini_embeddings(texts_own)
                        if temp_emb_own.size > 0:
                            st.session_state["emb_own"] = temp_emb_own
                            st.success("🎉 Eigene Liste erfolgreich verarbeitet!")
                            st.rerun()
                        else:
                            st.error("❌ Fehler bei der Berechnung von Liste 2.")

            # --- SCHRITT 3: DER FINALE VERGLEICH ---
            if st.session_state["emb_ext"] is not None and st.session_state["emb_own"] is not None:
                st.success("🎉 Alle Daten liegen stabil im Speicher bereit!")
                if st.button("⚡ Semantischen Vergleich berechnen", key="btn_schritt3"):
                    emb_ext = st.session_state["emb_ext"]
                    emb_own = st.session_state["emb_own"]
                    
                    with st.spinner("Berechne Ähnlichkeits-Matrix..."):
                        norm_ext = emb_ext / np.linalg.norm(emb_ext, axis=1, keepdims=True)
                        norm_own = emb_own / np.linalg.norm(emb_own, axis=1, keepdims=True)
                        
                        similarity_matrix = np.dot(norm_ext, norm_own.T)
                        best_own_indices = np.argmax(similarity_matrix, axis=1)
                        best_scores = np.max(similarity_matrix, axis=1)
                        
                        results = []
                        for idx_ext, best_idx_own in enumerate(best_own_indices):
                            score_percent = round(best_scores[idx_ext] * 100, 1)
                            if score_percent >= score_threshold:
                                results.append({
                                    "Externes Patent": df_ext.iloc[idx_ext]['Patentnummer'], 
                                    "Titel (Extern)": df_ext.iloc[idx_ext]['Titel_Uebersetzt'], 
                                    "Ähnlichstes eigenes Patent": df_own.iloc[best_idx_own]['Patentnummer'], 
                                    "Titel (Eigen)": df_own.iloc[best_idx_own]['Titel_Uebersetzt'], 
                                    "Match Score": f"{score_percent} %"
                                })
                        
                        if results: 
                            df_res = pd.DataFrame(results)
                            df_res['sort_val'] = df_res['Match Score'].str.replace(' %', '').astype(float)
                            df_res = df_res.sort_values(by='sort_val', ascending=False).drop(columns=['sort_val'])
                            st.dataframe(df_res, use_container_width=True)
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
