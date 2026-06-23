import streamlit as st
import pandas as pd
import numpy as np
import requests
import base64
import xml.etree.ElementTree as ET
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
    key = st.secrets["EPA_CONSUMER_KEY"]
    secret = st.secrets["EPA_CONSUMER_SECRET"]
    credential_bytes = f"{key}:{secret}".encode('utf-8')
    credential_base64 = base64.b64encode(credential_bytes).decode('utf-8')
    
    url = "https://ops.epo.org/3.2/auth/accesstoken"
    headers = {"Authorization": f"Basic {credential_base64}", "Content-Type": "application/x-www-form-urlencoded"}
    try:
        response = requests.post(url, headers=headers, data={"grant_type": "client_credentials"})
        return response.json().get("access_token") if response.status_code == 200 else None
    except:
        return None

def search_epa_keywords(query_string, token):
    """Sucht nach Stichwörtern im EPA und holt über mehrere Seiten bis zu 100 Treffer."""
    patents_found = []
    ns = {'ops': 'http://ops.epo.org', 'exchange': 'http://www.epo.org/exchange'}
    
    # Wir fragen 4 Seiten ab (Treffer 1-25, 26-50, 51-75, 76-100)
    ranges = ["1-25", "26-50", "51-75", "76-100"]
    
    for r in ranges:
        # Das EPA verlangt die Range im Header oder in der URL
        url = f"https://ops.epo.org/3.2/rest-services/published-data/search?q=txt={query_string}"
        headers = {
            "Authorization": f"Bearer {token}",
            "X-OPS-Range": r
        }
        
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                root = ET.fromstring(response.text)
                
                page_items = 0
                for doc in root.findall('.//ops:publication-reference', ns):
                    country_elem = doc.find('.//exchange:country', ns)
                    doc_num_elem = doc.find('.//exchange:doc-number', ns)
                    kind_elem = doc.find('.//exchange:kind', ns)
                    
                    if doc_num_elem is not None and doc_num_elem.text:
                        country = country_elem.text if country_elem is not None else ""
                        num = doc_num_elem.text
                        kind = kind_elem.text if kind_elem is not None else ""
                        full_number = f"{country}{num}{kind}"
                        
                        espacenet_url = f"https://worldwide.espacenet.com/patent/search?q={full_number}"
                        
                        # Duplikate vermeiden
                        if not any(p['Patentnummer'] == full_number for p in patents_found):
                            patents_found.append({
                                "Patentnummer": full_number,
                                "Datenbank": "EPA (Espacenet)",
                                "Espacenet Link": espacenet_url
                            })
                            page_items += 1
                
                # Wenn eine Seite weniger als 25 Treffer liefert, gibt es keine weiteren Seiten mehr
                if page_items < 25:
                    break
            elif response.status_code == 404:
                # 404 bedeutet beim EPA oft einfach, dass es für diese Range keine Treffer mehr gibt
                break
            else:
                st.error(f"Fehler bei Reichweite {r}: {response.status_code}")
                break
        except Exception as e:
            st.error(f"Fehler beim Abruf von Seite {r}: {e}")
            break
            
    return patents_found

# --- APP STARTEN ---
if check_password():
    tab_vergleich, tab_suche = st.tabs(["📊 Patent-Listen Vergleich", "🔍 Live-Recherche (EPA & Web)"])

    # REITER 1 (Hier bleibt alles wie gehabt)
    with tab_vergleich:
        st.title("💡 Patent Analyse Tool (KI-Berechnung)")
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
            score_threshold = st.slider("Mindest-Score für Relevanz (in %)", 0, 100, 30)
            if st.button("Semantische Nähe berechnen"):
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
                        if sim > best_score:
                            best_score, best_id, best_title = sim, df_own.iloc[idx_own]['Patentnummer'], df_own.iloc[idx_own]['Titel_Uebersetzt']
                    if round(best_score * 100, 1) >= score_threshold:
                        results.append({"Externes Patent": df_ext.iloc[idx_ext]['Patentnummer'], "Titel (Extern)": df_ext.iloc[idx_ext]['Titel_Uebersetzt'], "Ähnlichstes eigenes Patent": best_id, "Titel (Eigen)": best_title, "Match Score": f"{round(best_score * 100, 1)} %"})
                if results: st.dataframe(pd.DataFrame(results), use_container_width=True)

    # REITER 2 (Erweitert um Multi-Such-Logik und Paginierung)
    with tab_suche:
        st.title("🔍 Live Patent-Recherche & Stand der Technik")
        search_type = st.radio("Wonach möchtest du suchen?", ["Patentnummer eingeben", "Stichwörter (Stand der Technik)"])

        if search_type == "Patentnummer eingeben":
            st.subheader("Ähnliche Patente zu einer Nummer finden")
            patent_input = st.text_input("Patentnummer eingeben (z.B. EP3000000):")
            if st.button("Ähnliche Patente suchen"):
                if patent_input:
                    direct_url = f"https://worldwide.espacenet.com/patent/search?q={patent_input}"
                    st.markdown(f"👉 [Klicke hier, um das Patent direkt auf Espacenet zu prüfen]({direct_url})")
        else:
            st.subheader("Stand der Technik über Stichwörter ermitteln")
            
            # Info-Box für den Nutzer, wie man präzise sucht
            st.info("""
            💡 **Tipps für eine präzise Suche mit mehreren Begriffen:**
            * Nutzen Sie **`AND`** (Großbuchstaben), um Begriffe zu verknüpfen: z.B. `battery AND lithium AND drone`
            * Nutzen Sie **`OR`**, wenn alternative Begriffe erlaubt sind: z.B. `drone OR 'unmanned aerial vehicle'`
            * Nutzen Sie einfache Anführungszeichen **`'`** für exakte Wortgruppen: z.B. `'solid state battery'`
            """)
            
            keywords_input = st.text_input("Suchbegriffe eingeben (Englisch erforderlich):", value="battery AND lithium")
            
            if st.button("EPA Datenbank live durchsuchen"):
                if not keywords_input:
                    st.error("Bitte gib Suchbegriffe ein.")
                else:
                    with st.spinner("Frage EPA-Datenbank ab (holt bis zu 100 Treffer)..."):
                        token = get_epa_token()
                        if token:
                            results = search_epa_keywords(keywords_input, token)
                            if results:
                                st.success(f"Erfolgreich {len(results)} Patente beim EPA gefunden (max. 100 voreingestellt)!")
                                df_live = pd.DataFrame(results)
                                st.data_editor(df_live, column_config={"Espacenet Link": st.column_config.LinkColumn("Link zu Espacenet", display_text="↗ In Espacenet öffnen")}, disabled=True, use_container_width=True)
                                
                                csv_live = df_live.to_csv(index=False).encode('utf-8')
                                st.download_button(label="Suchergebnisse als CSV herunterladen", data=csv_live, file_name="epa_suchergebnisse.csv", mime="text/csv")
                            else:
                                st.warning("Keine Treffer gefunden. Überprüfe die Schreibweise oder vereinfache die Schlagwörter.")
