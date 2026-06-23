import streamlit as st
import pandas as pd
import numpy as np
import requests
import base64
import time
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

def fetch_patent_details(country, num, kind, token):
    """Holt Titel und Abstract (Zusammenfassung) für eine konkrete Patentnummer vom EPA."""
    # Wir nutzen den biblio-Endpunkt für Titel und den abstract-Endpunkt für die Zusammenfassung
    url = f"https://ops.epo.org/3.2/rest-services/published-data/publication/epodoc/{country}{num}.{kind}/biblio,abstract"
    headers = {"Authorization": f"Bearer {token}"}
    
    title = "Kein Titel im Index"
    abstract = "Keine Zusammenfassung im Index"
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            root = ET.fromstring(response.text)
            ns = {
                'exchange': 'http://www.epo.org/exchange',
                'ops': 'http://ops.epo.org'
            }
            
            # 1. Englischen Titel suchen
            titles = root.findall('.//exchange:title', ns)
            if titles:
                title = titles[0].text if titles[0].text else title
                for t in titles:
                    if t.get('lang') == 'en' and t.text:
                        title = t.text
                        break
                        
            # 2. Englischen Abstract suchen
            abstracts = root.findall('.//exchange:abstract', ns)
            if abstracts:
                abstract = abstracts[0].find('.//exchange:p', ns).text if abstracts[0].find('.//exchange:p', ns) is not None else abstract
                for a in abstracts:
                    if a.get('lang') == 'en':
                        p_elem = a.find('.//exchange:p', ns)
                        if p_elem is not None and p_elem.text:
                            abstract = p_elem.text
                            break
        return title, abstract
    except:
        return title, abstract

def search_epa_and_analyze(query_string, filter_criterion, score_threshold, token, model):
    """Sucht Patente, holt deren Texte und filtert sie live mit der KI."""
    patents_found = []
    ns = {'ops': 'http://ops.epo.org', 'exchange': 'http://www.epo.org/exchange'}
    
    # Aus Sicherheitsgründen fragen wir hier erst einmal die ersten 50 Treffer ab (2 Seiten)
    ranges = ["1-25", "26-50"]
    raw_numbers = []
    
    # SCHRITT 1: Nummern holen
    for r in ranges:
        url = f"https://ops.epo.org/3.2/rest-services/published-data/search?q=txt={query_string}"
        headers = {"Authorization": f"Bearer {token}", "X-OPS-Range": r}
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                root = ET.fromstring(response.text)
                items_count = 0
                for doc in root.findall('.//ops:publication-reference', ns):
                    c = doc.find('.//exchange:country', ns)
                    n = doc.find('.//exchange:doc-number', ns)
                    k = doc.find('.//exchange:kind', ns)
                    if n is not None and n.text:
                        country = c.text if c is not None else ""
                        num = n.text
                        kind = k.text if k is not None else ""
                        if (country, num, kind) not in raw_numbers:
                            raw_numbers.append((country, num, kind))
                            items_count += 1
                if items_count < 25:
                    break
            elif response.status_code == 404:
                break
        except:
            break

    if not raw_numbers:
        return []

    # SCHRITT 2: Texte vom EPA abrufen & KI-Abgleich durchführen
    progress_bar = st.progress(0, text="Rufe Patenttexte ab und starte KI-Filterung...")
    total_patents = len(raw_numbers)
    
    # Das Wunsch-Kriterium des Nutzers vektorisieren
    criterion_embedding = model.encode(filter_criterion, convert_to_numpy=True)
    
    for idx, (country, num, kind) in enumerate(raw_numbers):
        # Fortschrittsanzeige aktualisieren
        progress_text = f"Analysiere Patent {idx+1} von {total_patents} ({country}{num})..."
        progress_bar.progress((idx + 1) / total_patents, text=progress_text)
        
        # Detail-Texte (Titel, Abstract) von der API holen
        title, abstract = fetch_patent_details(country, num, kind, token)
        
        # KI-Abgleich: Text des Patents zusammenfügen
        patent_text = f"{title} {abstract}"
        patent_embedding = model.encode(patent_text, convert_to_numpy=True)
        
        # Cosinus-Ähnlichkeit berechnen
        sim = np.dot(criterion_embedding, patent_embedding) / (np.linalg.norm(criterion_embedding) * np.linalg.norm(patent_embedding))
        percentage_score = round(sim * 100, 1)
        
        # Nur aufnehmen, wenn der Schwellenwert erreicht wurde
        if percentage_score >= score_threshold:
            full_number = f"{country}{num}{kind}"
            espacenet_url = f"https://worldwide.espacenet.com/patent/search?q={full_number}"
            
            patents_found.append({
                "Patentnummer": full_number,
                "Titel": title,
                "Zusammenfassung (Abstract)": abstract if len(abstract) < 150 else abstract[:150] + "...",
                "KI Relevanz Score": f"{percentage_score} %",
                "Espacenet Link": espacenet_url,
                "raw_score": percentage_score  # Für die Sortierung
            })
            
        # Eine winzige Pause einbauen, um das API-Limit des EPA (max 60/min) absolut sicher einzuhalten
        time.sleep(0.3)
        
    progress_bar.empty()
    
    # Nach Relevanz sortieren
    if patents_found:
        df_temp = pd.DataFrame(patents_found)
        df_temp = df_temp.sort_values(by="raw_score", ascending=False).drop(columns=["raw_score"])
        return df_temp.to_dict('records')
    return []

# --- APP STARTEN ---
if check_password():
    tab_vergleich, tab_suche = st.tabs(["📊 Patent-Listen Vergleich", "🔍 Live-Recherche (EPA & KI-Filter)"])

    # REITER 1 (Listen-Vergleich)
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
            score_threshold = st.slider("Mindest-Score für Relevanz (in %)", 0, 100, 30, key="slider1")
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
                        if sim > best_score: best_score, best_id, best_title = sim, df_own.iloc[idx_own]['Patentnummer'], df_own.iloc[idx_own]['Titel_Uebersetzt']
                    if round(best_score * 100, 1) >= score_threshold:
                        results.append({"Externes Patent": df_ext.iloc[idx_ext]['Patentnummer'], "Titel (Extern)": df_ext.iloc[idx_ext]['Titel_Uebersetzt'], "Ähnlichstes eigenes Patent": best_id, "Titel (Eigen)": best_title, "Match Score": f"{round(best_score * 100, 1)} %"})
                if results: st.dataframe(pd.DataFrame(results), use_container_width=True)

    # REITER 2 (Live-Recherche MIT NEUEM KI-FILTER UND FREITEXT)
    with tab_suche:
        st.title("🔍 Live Patent-Recherche & KI-Filterung")
        st.write("Suche live im EPA und filtere die Ergebnisse sofort nach deinen inhaltlichen Vorgaben.")

        # Das KI-Modell auch hier bereitstellen
        @st.cache_resource
        def load_local_model_suche(): return SentenceTransformer('all-MiniLM-L6-v2')
        model_suche = load_local_model_suche()

        # Such-Eingaben
        col_stichworte, col_kriterium = st.columns(2)
        
        with col_stichworte:
            st.subheader("1. Grobe EPA-Vorauswahl (Datenbank)")
            keywords_input = st.text_input("Suchbegriffe für die Datenbank (Englisch, z.B. `battery AND drone`):", value="battery AND drone")
            
        with col_kriterium:
            st.subheader("2. Feiner KI-Relevanzfilter (Inhalt)")
            filter_input = st.text_input("Was genau interessiert dich an diesen Patenten? (Filter-Kriterium, z.B.: `cooling systems or heat management`):", value="cooling systems or heat management")

        st.markdown("---")
        st.subheader("🤖 KI-Filter Einstellungen")
        live_score_threshold = st.slider("Mindest-Übereinstimmung für Relevanz (in %)", min_value=0, max_value=100, value=25, key="slider2")
        
        if st.button("EPA live durchsuchen & mit KI filtern"):
            if not keywords_input or not filter_input:
                st.error("Bitte fülle sowohl die Suchbegriffe als auch das KI-Filter-Kriterium aus.")
            else:
                with st.spinner("Melde beim Europäischen Patentamt an..."):
                    token = get_epa_token()
                    
                    if token:
                        # Aufruf der kombinierten Suche, Abruf- und Analyselogik
                        analyzed_results = search_epa_and_analyze(keywords_input, filter_input, live_score_threshold, token, model_suche)
                        
                        if analyzed_results:
                            st.success(f"Analyse abgeschlossen! {len(analyzed_results)} Patente erfüllen dein KI-Relevanzkriterium von >= {live_score_threshold}%.")
                            
                            df_live_analyzed = pd.DataFrame(analyzed_results)
                            
                            # Tabelle perfekt formatiert ausgeben
                            st.data_editor(
                                df_live_analyzed,
                                column_config={
                                    "Espacenet Link": st.column_config.LinkColumn(
                                        "Link zu Espacenet",
                                        display_text="↗ In Espacenet öffnen"
                                    )
                                },
                                disabled=True,
                                use_container_width=True
                            )
                            
                            # Export-Möglichkeit der vorsortierten Liste
                            csv_live = df_live_analyzed.to_csv(index=False).encode('utf-8')
                            st.download_button(label="Relevante Patente als CSV herunterladen", data=csv_live, file_name="ki_gefilterte_patente.csv", mime="text/csv")
                        else:
                            st.warning("Keine Patente gefunden, die das Relevanz-Kriterium bei diesem Prozentwert erreichen. Drehe den Mindest-Score etwas nach unten oder passe das Kriterium an.")
