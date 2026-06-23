import streamlit as st
import pandas as pd
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# Seiteneinstellungen
st.set_page_config(page_title="Patent Analyse Tool", layout="wide")

# --- PASSTWORT SCHUTZ FUNKTION ---
def check_password():
    if "password_correct" not in st.session_state:
        st.title("🔒 Login erforderlich")
        st.text_input("Bitte gib das Passwort ein:", type="password", on_change=lambda: st.session_state.update({"password_correct": st.session_state["password"] == st.secrets["APP_PASSWORD"]}), key="password")
        return False
    return st.session_state["password_correct"]

# --- OPENALEX API HILFSFUNKTION ---
def search_openalex_patents(query_string, filter_criterion, score_threshold, model, max_results=100):
    """Sucht Patente via OpenAlex, holt Texte in einem Rutsch und filtert sie blitzschnell mit der KI."""
    # OpenAlex API URL für Patente (Typ: 'patent')
    # Wir suchen in den Titeln und Abstracts nach den eingegebenen Stichwörtern
    url = f"https://api.openalex.org/works?filter=type:patent,title_and_abstract.search:{query_string}&per_page={max_results}"
    
    patents_found = []
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            
            if not results:
                return []
                
            # KI-Vektor für das gewünschte Filter-Kriterium berechnen
            criterion_embedding = model.encode(filter_criterion, convert_to_numpy=True)
            
            # Da OpenAlex alles liefert, brauchen wir keine Zwangspause! Wir loopen direkt durch.
            for work in results:
                title = work.get("title", "Kein Titel verfügbar")
                
                # OpenAlex speichert Abstracts in einem speziellen Format (Inverted Index). 
                # Hier rekonstruieren wir den echten Text daraus:
                abstract_inverted = work.get("abstract_inverted")
                abstract = "Keine Zusammenfassung verfügbar"
                if abstract_inverted:
                    try:
                        abstract_words = {}
                        for word, positions in abstract_inverted.items():
                            for pos in positions:
                                abstract_words[pos] = word
                        abstract = " ".join([abstract_words[p] for p in sorted(abstract_words.keys())])
                    except:
                        pass
                
                # Patentnummer extrahieren (liegt meistens im Feld 'ids' oder 'display_name')
                # OpenAlex nutzt oft das Format 'Patent: US-123456-A1'
                display_name = work.get("display_name", "")
                patent_id = display_name.replace("Patent: ", "") if "Patent:" in display_name else display_name
                if not patent_id:
                    # Fallback falls display_name leer ist
                    patent_id = work.get("id", "").split("/")[-1]

                # Link generieren (OpenAlex bietet oft direkte Links, ansonsten nutzen wir Espacenet als Standard)
                clean_num_for_link = patent_id.replace("-", "").replace(" ", "")
                espacenet_url = f"https://worldwide.espacenet.com/patent/search?q={clean_num_for_link}"
                
                # --- KI ANALYSE ---
                patent_text = f"{title} {abstract}"
                patent_embedding = model.encode(patent_text, convert_to_numpy=True)
                
                # Cosinus-Ähnlichkeit berechnen
                sim = np.dot(criterion_embedding, patent_embedding) / (np.linalg.norm(criterion_embedding) * np.linalg.norm(patent_embedding))
                percentage_score = round(sim * 100, 1)
                
                # Filter anwenden
                if percentage_score >= score_threshold:
                    patents_found.append({
                        "Patentnummer": patent_id,
                        "Titel": title,
                        "Zusammenfassung (Abstract)": abstract if len(abstract) < 150 else abstract[:150] + "...",
                        "KI Relevanz Score": f"{percentage_score} %",
                        "Link zur Quelle": espacenet_url,
                        "raw_score": percentage_score
                    })
            
            # Ergebnisse nach Relevanz sortieren
            if patents_found:
                df_temp = pd.DataFrame(patents_found)
                df_temp = df_temp.sort_values(by="raw_score", ascending=False).drop(columns=["raw_score"])
                return df_temp.to_dict('records')
                
        return []
    except Exception as e:
        st.error(f"Fehler bei der OpenAlex-Abfrage: {e}")
        return []

# --- APP START ---
if check_password():
    tab_vergleich, tab_suche = st.tabs(["📊 Patent-Listen Vergleich", "🔍 Live-Recherche (OpenAlex Massen-Filter)"])

    # =========================================================================
    # REITER 1: PATENT-LISTEN VERGLEICH (TEIL 1)
    # =========================================================================
    with tab_vergleich:
        st.title("💡 Patent Analyse Tool (KI-Berechnung)")
        st.write("Lade zwei Excel-Listen (.xlsx oder .xlsm) hoch, um sie auf technische Nähe zu prüfen.")

        @st.cache_resource
        def load_local_model(): return SentenceTransformer('all-MiniLM-L6-v2')
        model = load_local_model()
        
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
            score_threshold = st.slider("Mindest-Score für Relevanz (in %)", 0, 100, 30, key="slider1")
            if st.button("Semantische Nähe berechnen"):
                with st.spinner("Analyse läuft..."):
                    texts_ext = (df_ext['Titel_Uebersetzt'].astype(str) + " " + df_ext['Zusammenfassung_Uebersetzt'].astype(str)).tolist()
                    texts_own = (df_own['Titel_Uebersetzt'].astype(str) + " " + df_own['Zusammenfassung_Uebersetzt'].astype(str)).tolist()
                    emb_ext = model.encode(texts_ext, convert_to_numpy=True)
                    emb_own = model.encode(texts_own, convert_to_numpy=True)
                    results = []
                    for idx_ext, e_ext in enumerate(emb_ext):
                        best_score = 0
                        best_id, best_title = "", ""
                        for idx_own, e_own in enumerate(emb_own):
                            sim = np.dot(e_ext, e_own) / (np.linalg.norm(e_ext) * np.linalg.norm(e_own))
                            if sim > best_score: best_score, best_id, best_title = sim, df_own.iloc[idx_own]['Patentnummer'], df_own.iloc[idx_own]['Titel_Uebersetzt']
                        if round(best_score * 100, 1) >= score_threshold:
                            results.append({"Externes Patent": df_ext.iloc[idx_ext]['Patentnummer'], "Titel (Extern)": df_ext.iloc[idx_ext]['Titel_Uebersetzt'], "Ähnlichstes eigenes Patent": best_id, "Titel (Eigen)": best_title, "Match Score": f"{round(best_score * 100, 1)} %"})
                    if results: st.dataframe(pd.DataFrame(results), use_container_width=True)

    # =========================================================================
    # REITER 2: LIVE-RECHERCHE (JETZT MIT UNLIMITIERTEM OPENALEX BULK-DOWNLOAD)
    # =========================================================================
    with tab_suche:
        st.title("🔍 Unlimitierte Live-Recherche & KI-Massenfilter")
        st.write("Durchsucht Millionen von weltweiten Patenten über OpenAlex in Sekundenschnelle ohne API-Limits.")

        @st.cache_resource
        def load_local_model_suche(): return SentenceTransformer('all-MiniLM-L6-v2')
        model_suche = load_local_model_suche()

        col_stichworte, col_kriterium = st.columns(2)
        with col_stichworte:
            st.subheader("1. Datenbank-Abfrage")
            keywords_input = st.text_input("Grobe Stichworte (Englisch, z.B. `solid state battery`):", value="solid state battery")
        with col_kriterium:
            st.subheader("2. KI-Feinfilter (Freitext)")
            filter_input = st.text_input("Worauf soll die KI filtern? (z.B. `anode materials or lithium metal silicon`):", value="anode materials")

        # Zusätzliche Einstellungen für die Massen-Analyse
        st.markdown("---")
        col_slider, col_max = st.columns([3, 1])
        with col_slider:
            live_score_threshold = st.slider("Mindest-Übereinstimmung für Relevanz (in %)", min_value=0, max_value=100, value=25, key="slider2")
        with col_max:
            max_results_input = st.selectbox("Wie viele Patente scannen?", [25, 50, 100, 200], index=2)

        if st.button("Massen-Suche & KI-Analyse starten"):
            if not keywords_input or not filter_input:
                st.error("Bitte fülle alle Textfelder aus.")
            else:
                with st.spinner(f"Frage OpenAlex ab und jage bis zu {max_results_input} Patente durch die KI..."):
                    
                    # Suche und Filterung starten
                    analyzed_results = search_openalex_patents(keywords_input, filter_input, live_score_threshold, model_suche, max_results=max_results_input)
                    
                    if analyzed_results:
                        st.success(f"Analyse blitzschnell beendet! {len(analyzed_results)} relevante Patente gefunden.")
                        df_live_analyzed = pd.DataFrame(analyzed_results)
                        
                        # Tabelle formatiert anzeigen
                        st.data_editor(
                            df_live_analyzed,
                            column_config={
                                "Link zur Quelle": st.column_config.LinkColumn(
                                    "Link zur Quelle",
                                    display_text="↗ Patent öffnen"
                                )
                            },
                            disabled=True,
                            use_container_width=True
                        )
                        
                        csv_live = df_live_analyzed.to_csv(index=False).encode('utf-8')
                        st.download_button(label="Gefilterte Patente als CSV herunterladen", data=csv_live, file_name="openalex_ki_treffer.csv", mime="text/csv")
                    else:
                        st.warning("Keine Patente gefunden, die diesen KI-Score erreichen. Verändere deine Suchwörter oder senke den Mindest-Score.")
