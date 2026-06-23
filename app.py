import streamlit as st
import pandas as pd
import numpy as np
import requests
import base64
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer

# Seiteneinstellungen (Muss der allererste Streamlit-Befehl sein!)
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
        st.error("❌ Falsches Passwort. Bitte versuche es erneut.")
        return False
    else:
        return True

# --- EPA API HILFSFUNKTIONEN ---
def get_epa_token():
    """Holt ein temporäres Access Token vom EPA OPS Service ab."""
    key = st.secrets["EPA_CONSUMER_KEY"]
    secret = st.secrets["EPA_CONSUMER_SECRET"]
    
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
    """Sucht nach Stichwörtern im EPA und liefert die Nummern und korrekten Links."""
    url = f"https://ops.epo.org/3.2/rest-services/published-data/search?q=txt={query_string}"
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            root = ET.fromstring(response.text)
            
            ns = {
                'ops': 'http://ops.epo.org',
                'exchange': 'http://www.epo.org/exchange'
            }
            
            patents_found = []
            
            # Alle Nummern auslesen
            for doc in root.findall('.//ops:publication-reference', ns):
                country_elem = doc.find('.//exchange:country', ns)
                doc_num_elem = doc.find('.//exchange:doc-number', ns)
                kind_elem = doc.find('.//exchange:kind', ns)
                
                if doc_num_elem is not None and doc_num_elem.text:
                    country = country_elem.text if country_elem is not None else ""
                    num = doc_num_elem.text
                    kind = kind_elem.text if kind_elem is not None else ""
                    
                    full_number = f"{country}{num}{kind}"
                    
                    # Offizielles Such-Link-Format für Espacenet
                    espacenet_url = f"https://worldwide.espacenet.com/patent/search?q={full_number}"
                    
                    patents_found.append({
                        "Patentnummer": full_number,
                        "Datenbank": "EPA (Espacenet)",
                        "Espacenet Link": espacenet_url
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

    # Haupt-Navigation über Tabs/Reiter
    tab_vergleich, tab_suche = st.tabs(["📊 Patent-Listen Vergleich", "🔍 Live-Recherche (EPA & Web)"])

    # =========================================================================
    # REITER 1: PATENT-LISTEN VERGLEICH (TEIL 1)
    # =========================================================================
    with tab_vergleich:
        st.title("💡 Patent Analyse Tool (KI-Berechnung)")
        st.write("Lade zwei Excel-Listen (.xlsx oder .xlsm) hoch, um sie auf technische Nähe zu prüfen.")

        # Lokales KI-Modell laden (wird im Speicher behalten)
        @st.cache_resource
        def load_local_model():
            return SentenceTransformer('all-MiniLM-L6-v2')

        with st.spinner("Lade KI-Modell in den Speicher..."):
            model = load_local_model()

        # Layout für die Uploads
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("1. Externe Patente")
            uploaded_file_ext = st.file_uploader("Excel-Liste hochladen (Extern)", type=["xlsx", "xlsm"])

        with col2:
            st.subheader("2. Eigene Patente")
            uploaded_file_own = st.file_uploader("Excel-Liste hochladen (Eigene)", type=["xlsx", "xlsm"])

        # Funktion zum sicheren Einlesen der Excel
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
                    st.error(f"Fehler beim Laden der Datei: {e}")
                    return None
            return None

        df_ext = load_patent_data(uploaded_file_ext)
        df_own = load_patent_data(uploaded_file_own)

        # Wenn beide Listen da sind, starten wir die Analyse-Optionen
        if df_ext is not None and df_own is not None:
            st.success("Beide Listen erfolgreich geladen!")
            
            st.markdown("---")
            st.subheader("🤖 KI-Analyse Einstellungen")
            score_threshold = st.slider("Mindest-Score für Relevanz (in %)", min_value=0, max_value=100, value=30)
            
            if st.button("Semantische Nähe berechnen"):
                with st.spinner("KI analysiert die Patente... Bitte warten..."):
                    
                    # Texte vorbereiten
                    texts_ext = (df_ext['Titel_Uebersetzt'].astype(str) + " " + df_ext['Zusammenfassung_Uebersetzt'].astype(str)).tolist()
                    texts_own = (df_own['Titel_Uebersetzt'].astype(str) + " " + df_own['Zusammenfassung_Uebersetzt'].astype(str)).tolist()
                    
                    # Vektoren lokal berechnen
                    embeds_ext = model.encode(texts_ext, convert_to_numpy=True)
                    embeds_own = model.encode(texts_own, convert_to_numpy=True)
                    
                    results = []
                    
                    # Abgleich durchführen
                    for idx_ext, emb_ext in enumerate(embeds_ext):
                        best_score = 0
                        best_match_own_id = ""
                        best_match_own_title = ""
                        
                        for idx_own, emb_own in enumerate(embeds_own):
                            # Cosinus-Ähnlichkeit berechnen
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
                    
                    st.markdown("---")
                    st.subheader("📋 Analyse-Ergebnisse")
                    
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
                        st.info("Keine Treffer über dem Mindest-Score gefunden.")

    # =========================================================================
    # REITER 2: LIVE-RECHERCHE (TEIL 2)
    # =========================================================================
    with tab_suche:
        st.title("🔍 Live Patent-Recherche & Stand der Technik")
        st.write("Durchsuche die offiziellen Live-Datenbanken des EPA über die OPS-Schnittstelle.")

        search_type = st.radio("Wonach möchtest du suchen?", ["Patentnummer eingeben", "Stichwörter (Stand der Technik)"])

        if search_type == "Patentnummer eingeben":
            st.subheader("Ähnliche Patente zu einer Nummer finden")
            patent_input = st.text_input("Patentnummer eingeben (z.B. EP3000000):")
            
            if st.button("Ähnliche Patente suchen"):
                if not patent_input:
                    st.error("Bitte gib eine Patentnummer ein.")
                else:
                    st.info(f"Suche nach ähnlichen Patenten für {patent_input}...")
                    # Direktverlinkung zu Espacenet für diese Nummer als schnelle Lösung
                    direct_url = f"https://worldwide.espacenet.com/patent/search?q={patent_input}"
                    st.markdown(f"👉 [Klicke hier, um das Patent direkt auf Espacenet zu prüfen]({direct_url})")

        else:
            st.subheader("Stand der Technik über Stichwörter ermitteln")
            keywords_input = st.text_input("Stichwörter eingeben (Verwende einfache Anführungszeichen für Wortgruppen, z.B. 'solid state battery'):")
            
            if st.button("EPA Datenbank live durchsuchen"):
                if not keywords_input:
                    st.error("Bitte gib Suchbegriffe ein.")
                else:
                    with st.spinner("Verbinde mit dem Europäischen Patentamt..."):
                        token = get_epa_token()
                        
                        if token:
                            st.write("🔄 Authentifizierung erfolgreich. Rufe Daten ab...")
                            results = search_epa_keywords(keywords_input, token)
                            
                            if results:
                                st.success(f"Erfolgreich {len(results)} Patente beim EPA gefunden!")
                                df_live = pd.DataFrame(results)
                                
                                # Anzeige der Tabelle mit funktionierenden klickbaren Links
                                st.data_editor(
                                    df_live,
                                    column_config={
                                        "Espacenet Link": st.column_config.LinkColumn(
                                            "Link zu Espacenet",
                                            display_text="↗ In Espacenet öffnen"
                                        )
                                    },
                                    disabled=True,
                                    use_container_width=True
                                )
                                
                                # Download Button für die Suchtreffer
                                csv_live = df_live.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="Suchergebnisse als CSV herunterladen",
                                    data=csv_live,
                                    file_name="epa_suchergebnisse.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.warning("Keine Treffer gefunden. Versuche ein anderes, englisches Stichwort.")
