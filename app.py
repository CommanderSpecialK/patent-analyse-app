import streamlit as st
import pandas as pd
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# Seiteneinstellungen
st.set_page_config(page_title="Patent Analyse Tool", layout="wide")

# --- PASSTWORT SCHUTZ FUNKTION ---
def check_password():
    """Gibt True zurück, wenn der Benutzer das richtige Passwort eingegeben hat."""
    def password_entered():
        if st.session_state["password"] == st.secrets["APP_PASSWORD"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.title("🔒 Login erforderlich")
        st.text_input("Bitte gib das Passwort ein:", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.title("🔒 Login erforderlich")
        st.text_input("Bitte gib das Passwort ein:", type="password", on_change=password_entered, key="password")
        st.error("❌ Falsches Passwort.")
        return False
    else:
        return True

# --- APP STARTEN, WENN PASSWORT KORREKT ---
if check_password():

    # Haupt-Navigation über Tabs
    tab_vergleich, tab_suche = st.tabs(["📊 Patent-Listen Vergleich", "🔍 Live-Recherche (EPA & Web)"])

    # =========================================================================
    # REITER 1: PATENT-LISTEN VERGLEICH (DEIN FERTIGER TEIL 1)
    # =========================================================================
    with tab_vergleich:
        st.title("💡 Patent Analyse Tool (KI-Berechnung)")
        st.write("Lade zwei Excel-Listen (.xlsx oder .xlsm) hoch, um sie auf technische Nähe zu prüfen.")

        @st.cache_resource
        def load_local_model():
            return SentenceTransformer('all-MiniLM-L6-v2')

        with st.spinner("Lade KI-Modell in den Speicher..."):
            model = load_local_model()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("1. Externe Patente")
            uploaded_file_ext = st.file_uploader("Excel-Liste hochladen (Extern)", type=["xlsx", "xlsm"])
        with col2:
            st.subheader("2. Eigene Patente")
            uploaded_file_own = st.file_uploader("Excel-Liste hochladen (Eigene)", type=["xlsx", "xlsm"])

        def load_patent_data(uploaded_file):
            if uploaded_file is not None:
                try:
                    df = pd.read_excel(uploaded_file, engine="openpyxl")
                    if df.shape[1] < 4:
                        st.error("Die Datei muss mindestens 4 Spalten haben!")
                        return None
                    df.columns = ['Patentnummer', 'Titel_Original', 'Titel_Uebersetzt', 'Zusammenfassung_Uebersetzt'] + list(df.columns[4:])
                    df = df.fillna("")
                    return df
                except Exception as e:
                    st.error(f"Fehler beim Laden der Datei: {e}")
                    return None
            return None

        df_ext = load_patent_data(uploaded_file_ext)
        df_own = load_patent_data(uploaded_file_own)

        def cosine_similarity(v1, v2):
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

        if df_ext is not None and df_own is not None:
            st.success("Beide Listen erfolgreich geladen!")
            st.markdown("---")
            st.subheader("🤖 KI-Analyse Einstellungen")
            score_threshold = st.slider("Mindest-Score für Relevanz (in %)", min_value=0, max_value=100, value=30)
            
            if st.button("Semantische Nähe berechnen"):
                with st.spinner("KI analysiert die Patente... Bitte warten..."):
                    texts_ext = (df_ext['Titel_Uebersetzt'].astype(str) + " " + df_ext['Zusammenfassung_Uebersetzt'].astype(str)).tolist()
                    texts_own = (df_own['Titel_Uebersetzt'].astype(str) + " " + df_own['Zusammenfassung_Uebersetzt'].astype(str)).tolist()
                    
                    embeds_ext = model.encode(texts_ext, convert_to_numpy=True)
                    embeds_own = model.encode(texts_own, convert_to_numpy=True)
                    
                    results = []
                    for idx_ext, emb_ext in enumerate(embeds_ext):
                        best_score = 0
                        best_match_own_id = ""
                        best_match_own_title = ""
                        
                        for idx_own, emb_own in enumerate(embeds_own):
                            sim = cosine_similarity(emb_ext, emb_own)
                            if sim > best_score:
                                best_score = sim
                                best_match_own_id = df_own.iloc[idx_own]['Patentnummer']
                                best_match_own_title = df_own.iloc[idx_own]['Titel_Uebersetzt']
                        
                        percentage_score = round(best_score * 100, 1)
                        if percentage_score >= score_threshold:
                            results.append({
                                "Externes Patent": df_ext.iloc[idx_ext]['Patentnummer'],
                                "Titel (Extern)": df_ext.iloc[idx_ext]['Titel_Uebersetzt'],
                                "Ähnlichstes eigenes Patent": best_match_own_id,
                                "Titel (Eigen)": best_match_own_title,
                                "Match Score": f"{percentage_score} %"
                            })
                    
                    st.markdown("---")
                    st.subheader("📋 Analyse-Ergebnisse")
                    if results:
                        df_results = pd.DataFrame(results)
                        df_results['sort_col'] = df_results['Match Score'].str.replace(' %', '').astype(float)
                        df_results = df_results.sort_values(by='sort_col', ascending=False).drop(columns=['sort_col'])
                        st.dataframe(df_results, use_container_width=True)
                        
                        csv = df_results.to_csv(index=False).encode('utf-8')
                        st.download_button(label="Ergebnisse als CSV herunterladen", data=csv, file_name="patent_analyse_ergebnisse.csv", mime="text/csv")
                    else:
                        st.info("Keine Treffer über dem Mindest-Score gefunden.")

    # =========================================================================
    # REITER 2: LIVE-RECHERCHE (NEUER TEIL 2)
    # =========================================================================
    with tab_suche:
        st.title("🔍 Live Patent-Recherche & Stand der Technik")
        st.write("Durchsuche die offiziellen Datenbanken des EPA (Espacenet) und des Internets live.")

        # Auswahl der Suchart
        search_type = st.radio("Wonach möchtest du suchen?", ["Patentnummer eingeben", "Stichwörter (Stand der Technik)"])

        if search_type == "Patentnummer eingeben":
            st.subheader("Ähnliche Patente zu einer Nummer finden")
            patent_input = st.text_input("Patentnummer eingeben (z.B. EP3000000 oder US20210123456):")
            
            if st.button("Ähnliche Patente suchen"):
                if not patent_input:
                    st.error("Bitte gib eine Patentnummer ein.")
                else:
                    st.info(f"Suche nach Patentnummer {patent_input} gestartet...")
                    # HIER binden wir gleich die EPA API ein!
                    st.warning("Hier wird im nächsten Schritt die Live-Schnittstelle abgefragt.")

        else:
            st.subheader("Stand der Technik über Stichwörter ermitteln")
            keywords_input = st.text_input("Stichwörter / Suchbegriffe eingeben (Englisch wird empfohlen, z.B.: 'solid state battery lithium'):")
            
            if st.button("Datenbanken durchsuchen"):
                if not keywords_input:
                    st.error("Bitte gib mindestens ein Stichwort ein.")
                else:
                    st.info(f"Suche nach Begriffen '{keywords_input}' gestartet...")
                    # HIER binden wir gleich die Freitext-Suche ein!
                    st.warning("Hier wird im nächsten Schritt die Freitext-Suche abgefragt.")
