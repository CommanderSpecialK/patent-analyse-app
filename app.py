import streamlit as st
import pandas as pd
import numpy as np
import requests
import base64
import xml.etree.ElementTree as ET  # Das EPA sendet Daten im XML-Format, das wir damit lesen
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

# --- EPA API HILFSFUNKTIONEN ---
def get_epa_token():
    """Holt ein temporäres Access Token vom EPA OPS Service ab."""
    key = st.secrets["EPA_CONSUMER_KEY"]
    secret = st.secrets["EPA_CONSUMER_SECRET"]
    
    # Schlüssel müssen für das EPA in Base64 codiert werden
    credential_bytes = f"{key}:{secret}".encode('utf-8')
    credential_base64 = base64.b64encode(credential_bytes).decode('utf-8')
    
    url = "https://ops.epo.org/3.2/auth/accesstoken"
    headers = {
        "Authorization": f"Basic {credential_base64}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    payload = {"grant_type": "client_credentials"}
    
    try:
        response = requests.post(url, headers=headers, data=payload)
        if response.status_code == 200:
            return response.json().get("access_token")
        else:
            st.error(f"EPA Authentifizierungs-Fehler ({response.status_code}): {response.text}")
            return None
    except Exception as e:
        st.error(f"Verbindung zum EPA fehlgeschlagen: {e}")
        return None

def search_epa_keywords(query_string, token):
    """Sucht nach Stichwörtern in der EPA Datenbank (Espacenet)."""
    # CQL ist die Suchsprache des EPA. txt=Sucht in Titel/Abstract
    url = f"https://ops.epo.org/3.2/rest-services/published-data/search?q=txt={query_string}"
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            # Das EPA antwortet in XML. Wir extrahieren die wichtigsten Daten:
            root = ET.fromstring(response.text)
            
            # Namespaces, die das EPA im XML nutzt
            ns = {
                'ops': 'http://ops.epo.org',
                'exchange': 'http://www.epo.org/exchange',
                'ccd': 'http://www.epo.org/ccd'
            }
            
            patents_found = []
            
            # Wir schleifen durch alle gefundenen Dokumente (maximal 25 Ergebnisse standardmäßig)
            for doc in root.findall('.//ops:publication-reference', ns):
                doc_num_elem = doc.find('.//exchange:doc-number', ns)
                country_elem = doc.find('.//exchange:country', ns)
                kind_elem = doc.find('.//exchange:kind', ns)
                
                if doc_num_elem is not None:
                    country = country_elem.text if country_elem is not None else ""
                    num = doc_num_elem.text
                    kind = kind_elem.text if kind_elem is not None else ""
                    
                    full_number = f"{country}{num}{kind}"
                    patents_found.append({
                        "Patentnummer": full_number,
                        "Datenbank": "EPA (Espacenet)"
                    })
            return patents_found
        else:
            st.error(f"EPA Suche fehlgeschlagen ({response.status_code}): {response.text}")
            return []
    except Exception as e:
        st.error(f"Fehler bei der Live-Suche: {e}")
        return []

# --- APP STARTEN, WENN PASSWORT KORREKT ---
if check_password():

    tab_vergleich, tab_suche = st.tabs(["📊 Patent-Listen Vergleich", "🔍 Live-Recherche (EPA & Web)"])

    # =========================================================================
    # REITER 1: PATENT-LISTEN VERGLEICH (TEIL 1)
    # =========================================================================
    with tab_vergleich:
        st.title("💡 Patent Analyse Tool (KI-Berechnung)")
        st.write("Lade zwei Excel-Listen (.xlsx oder .xlsm) hoch, um sie auf technische Nähe zu prüfen.")

        @st.cache_resource
        def load_local_model():
            return SentenceTransformer('all-MiniLM-L6-v2')

        with st.spinner("Lade KI-Modell..."):
            model = load_local_model()

        col1, col2 = st.columns(2)
        with col1:
            uploaded_file_ext = st.file_uploader("Excel-Liste hochladen (Extern)", type=["xlsx", "xlsm"])
        with col2:
            uploaded_file_own = st.file_uploader("Excel-Liste hochladen (Eigene)", type=["xlsx", "xlsm"])

        def load_patent_data(uploaded_file):
            if uploaded_file is not None:
                try:
                    df = pd.read_excel(uploaded_file, engine="openpyxl")
                    if df.shape[1] < 4:
                        st.error("Die Datei muss mindestens 4 Spalten haben!")
                        return None
                    df.columns = ['Patentnummer', 'Titel_Original', 'Titel_Uebersetzt', 'Zusammenfassung_Uebersetzt'] + list(df.columns[4:])
                    return df.fillna("")
                except Exception as e:
                    st.error(f"Fehler beim Laden: {e}")
            return None

        df_ext = load_patent_data(uploaded_file_ext)
        df_own = load_patent_data(uploaded_file_own)

        if df_ext is not None and df_own is not None:
            st.success("Listen geladen!")
            score_threshold = st.slider("Mindest-Score für Relevanz (in %)", 0, 100, 30)
            
            if st.button("Semantische Nähe berechnen"):
                with st.spinner("Analyse läuft..."):
                    texts_ext = (df_ext['Titel_Uebersetzt'].astype(str) + " " + df_ext['Zusammenfassung_Uebersetzt'].astype(str)).tolist()
                    texts_own = (df_own['Titel_Uebersetzt'].astype(str) + " " + df_own['Zusammenfassung_Uebersetzt'].astype(str)).tolist()
                    
                    embeds_ext = model.encode(texts_ext, convert_to_numpy=True)
                    embeds_own = model.encode(texts_own, convert_to_numpy=True)
                    
                    results = []
                    for idx_ext, emb_ext in enumerate(embeds_ext):
                        best_score = 0
                        best_match_own_id, best_match_own_title = "", ""
                        for idx_own, emb_own in enumerate(embeds_own):
                            sim = np.dot(emb_ext, emb_own) / (np.linalg.norm(emb_ext) * np.linalg.norm(emb_own))
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
                    
                    if results:
                        df_results = pd.DataFrame(results).sort_values(by='Match Score', ascending=False)
                        st.dataframe(df_results, use_container_width=True)

    # =========================================================================
    # REITER 2: LIVE-RECHERCHE (TEIL 2 - JETZT AKTIV!)
    # =========================================================================
    with tab_suche:
        st.title("🔍 Live Patent-Recherche & Stand der Technik")
        st.write("Durchsuche die offiziellen Live-Datenbanken des EPA über die OPS-Schnittstelle.")

        search_type = st.radio("Wonach möchtest du suchen?", ["Patentnummer eingeben", "Stichwörter (Stand der Technik)"])

        if search_type == "Patentnummer eingeben":
            st.subheader("Ähnliche Patente zu einer Nummer finden")
            patent_input = st.text_input("Patentnummer eingeben (z.B. RE47539 oder 3000000):")
            
            if st.button("Ähnliche Patente suchen"):
                st.info("Funktion folgt im nächsten Teilschritt – wir testen zuerst die Stichwortsuche!")

        else:
            st.subheader("Stand der Technik über Stichwörter ermitteln")
            # Wichtig: Die EPA API versteht am besten englische Begriffe in einfachen Anführungszeichen
            keywords_input = st.text_input("Stichwörter eingeben (Verwende einfache Anführungszeichen für Wortgruppen, z.B. 'solid state battery'):")
            
            if st.button("EPA Datenbank live durchsuchen"):
                if not keywords_input:
                    st.error("Bitte gib Suchbegriffe ein.")
                else:
                    with st.spinner("Verbinde mit dem Europäischen Patentamt..."):
                        # 1. Token holen
                        token = get_epa_token()
                        
                        if token:
                            st.write("🔄 Authentifizierung erfolgreich. Starte Live-Abfrage...")
                            # 2. Suchen
                            results = search_epa_keywords(keywords_input, token)
                            
                            # 3. Ergebnisse anzeigen
                            if results:
                                st.success(f"Erfolgreich {len(results)} Patente beim EPA gefunden!")
                                df_live = pd.DataFrame(results)
                                st.dataframe(df_live, use_container_width=True)
                            else:
                                st.warning("Keine direkten Treffer gefunden. Versuche ein anderes, einzelnes englisches Stichwort (z.B. 'battery' oder 'sensor').")
