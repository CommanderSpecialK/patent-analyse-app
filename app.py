import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Seiteneinstellungen
st.set_page_config(page_title="Lokales Patent Analyse Tool", layout="wide")

st.title("💡 Lokales Patent Analyse Tool (100% Kostenlos & Offline)")
st.write("Lade zwei Excel-Listen (.xlsx oder .xlsm) hoch. Die Analyse läuft komplett lokal auf deinem PC.")

# Lokales KI-Modell laden (wird im Speicher behalten, damit es schnell bleibt)
@st.cache_resource
def load_local_model():
    # Ein extrem schnelles, kostenloses Modell für technische/semantische Nähe
    return SentenceTransformer('all-MiniLM-L6-v2')

with st.spinner("Lade KI-Modell in den Speicher... (Dauert nur beim ersten Start kurz)"):
    model = load_local_model()

# Layout für die Uploads
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Externe Patente")
    uploaded_file_ext = st.file_uploader("Excel-Liste hochladen (Extern)", type=["xlsx", "xlsm"])

with col2:
    st.subheader("2. Eigene Patente")
    uploaded_file_own = st.file_uploader("Excel-Liste hochladen (Eigene)", type=["xlsx", "xlsm"])

# Funktion zum sicheren Einlesen der Excel (unterstützt xlsx und xlsm)
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

# Hilfsfunktion: Berechnet die mathematische Nähe
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Wenn beide Listen da sind, starten wir
if df_ext is not None and df_own is not None:
    st.success("Beide Listen erfolgreich geladen!")
    
    st.markdown("---")
    st.subheader("🤖 KI-Analyse Einstellungen")
    score_threshold = st.slider("Mindest-Score für Relevanz (in %)", min_value=0, max_value=100, value=30)
    
    if st.button("Semantische Nähe berechnen"):
        with st.spinner("Lokale KI analysiert die Patente... Bitte warten..."):
            
            # Texte vorbereiten
            texts_ext = (df_ext['Titel_Uebersetzt'].astype(str) + " " + df_ext['Zusammenfassung_Uebersetzt'].astype(str)).tolist()
            texts_own = (df_own['Titel_Uebersetzt'].astype(str) + " " + df_own['Zusammenfassung_Uebersetzt'].astype(str)).tolist()
            
            # Vektoren lokal auf deinem PC berechnen (völlig unabhängig vom Internet)
            embeds_ext = model.encode(texts_ext, convert_to_numpy=True)
            embeds_own = model.encode(texts_own, convert_to_numpy=True)
            
            results = []
            
            # Abgleich durchführen (Jedes externe mit jedem eigenen vergleichen)
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
            
            # Ergebnisse anzeigen
            st.markdown("---")
            st.subheader("📋 Analyse-Ergebnisse (Lokale Berechnung)")
            
            if results:
                df_results = pd.DataFrame(results)
                df_results['sort_col'] = df_results['Match Score'].str.replace(' %', '').astype(float)
                df_results = df_results.sort_values(by='sort_col', ascending=False).drop(columns=['sort_col'])
                
                st.dataframe(df_results, use_container_width=True)
                
                csv = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Ergebnisse als CSV herunterladen",
                    data=csv,
                    file_name="patent_analyse_ergebnisse.csv",
                    mime="text/csv",
                )
            else:
                st.info("Keine Treffer über dem Mindest-Score gefunden. Schiebe den Regler weiter nach links.")