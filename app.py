

import feedparser

def fetch_rss_articles(query, max_results=10):
    articles = []
    feeds = [
        "https://news.google.com/rss/search?q=DS+Automobiles&hl=fr&gl=FR&ceid=FR:fr",
        "https://www.leblogauto.com/feed",
        "https://www.caradisiac.com/rss",
        "https://www.automobile-propre.com/feed/",
        "https://www.frandroid.com/feed",
        "https://www.largus.fr/rss",
        "https://www.auto-moto.com/feed"
    ]
    for url in feeds:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_results]:
                articles.append({
                    "date": entry.get("published", ""),
                    "titre": entry.get("title", ""),
                    "contenu": entry.get("summary", entry.get("description", "")),
                    "source": url.split("//")[1].split("/")[0],
                    "lien": entry.get("link", "")
                })
        except:
            pass
    return articles


import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import random
from transformers import pipeline

# --- CONFIGURATION ---
st.set_page_config(page_title="Veille DS Automobiles", layout="wide")
st.title("🚗 Agent de Veille – DS Automobiles (APIs multiples)")

traduire = st.checkbox("Traduire les contenus en français (résumé uniquement)")

API_KEY_NEWSDATA = st.secrets["API_KEY_NEWSDATA"]
MEDIASTACK_API_KEY = st.secrets["MEDIASTACK_API_KEY"]
SLACK_WEBHOOK_URL = st.secrets["SLACK_WEBHOOK_URL"]

NEWSDATA_URL = "https://newsdata.io/api/1/news"
MEDIASTACK_URL = "http://api.mediastack.com/v1/news"
RSS_FEEDS = [
    "https://news.google.com/rss/search?q=DS+Automobiles&hl=fr&gl=FR&ceid=FR:fr",
    "https://www.leblogauto.com/feed",
    "https://www.caradisiac.com/index.rss",
    "https://www.auto-moto.com/feed",
    "https://journalauto.com/feed",
    "https://www.automobile-propre.com/feed",
    "https://www.frandroid.com/feed",
    "https://www.largus.fr/rss/largus.xml"
]

MODELES_DS = ["DS N4", "DS N8", "DS7", "DS3", "DS9", "DS4", "N°8", "N°4", "Jules Verne"]

@st.cache_resource
def get_sentiment_pipeline():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
sentiment_analyzer = get_sentiment_pipeline()

# --- FONCTIONS DE COLLECTE ---
def fetch_newsdata_articles(query, max_results=15):
    params = {"apikey": API_KEY_NEWSDATA, "q": query}
    if langue and langue != "Tous":
        params["language"] = langue
    if pays and pays != "Tous":
        params["country"] = pays
    try:
        response = requests.get(NEWSDATA_URL, params=params)
        data = response.json()
        return [{
            "date": item.get("pubDate", ""),
            "titre": item.get("title", ""),
            "contenu": item.get("description", ""),
            "source": item.get("source_id", ""),
            "lien": item.get("link", "")
        } for item in data.get("results", [])[:max_results]]
    except:
        return []

def fetch_mediastack_articles(query, max_results=15):
    params = {"access_key": MEDIASTACK_API_KEY, "keywords": query}
    if langue and langue != "Tous":
        params["languages"] = langue
    if pays and pays != "Tous":
        params["countries"] = pays
    try:
        response = requests.get(MEDIASTACK_URL, params=params)
        data = response.json()
        return [{
            "date": item.get("published_at", ""),
            "titre": item.get("title", ""),
            "contenu": item.get("description", ""),
            "source": item.get("source", ""),
            "lien": item.get("url", "")
        } for item in data.get("data", [])[:max_results]]
    except:
        return []

def detecter_modele(titre):
    for m in MODELES_DS:
        if m.lower() in titre.lower():
            return m
    return "DS Global"

def analyser_article(row):
    contenu = row.get("contenu") or ""
    try:
        sentiment_result = sentiment_analyzer(contenu[:512])[0]
        sentiment_label = sentiment_result.get("label", "").upper()
        sentiment = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}.get(sentiment_label, sentiment_label)
    except:
        sentiment = "Neutral"
    modele = detecter_modele(row.get("titre") or "")
    résumé = contenu[:200] + "..." if contenu else "Aucun contenu"
    return pd.Series({"résumé": résumé, "ton": sentiment, "modèle": modele})

def envoyer_notif_slack(article):
    try:
        payload = {
            "text": f"📰 Nouvel article détecté sur *{article['modèle']}*\n"
                    f"*{article['titre']}*\n"
                    f"_Ton: {article['ton']}_\n"
                    f"<{article['lien']}|Lire l'article>"
        }
        requests.post(SLACK_WEBHOOK_URL, json=payload)
    except:
        pass

# --- INTERFACE UTILISATEUR ---
nb_jours = st.slider("Nombre de jours d'historique à afficher", 1, 90, 30)
nb_articles = st.slider("Nombre d'articles à récupérer (par source)", 5, 30, 10)
pays = st.selectbox("Filtrer par pays", ["Tous", "fr", "us", "gb", "de", "es", "it", "cn", "jp", "br", "nl"])
langue = st.selectbox("Filtrer par langue", ["Tous", "fr", "en", "es", "de", "it", "pt", "nl", "pl", "zh", "ja"])
filtre_modele = st.selectbox("Filtrer par modèle", ["Tous"] + MODELES_DS)
filtre_ton = st.selectbox("Filtrer par ton", ["Tous", "Positive", "Neutral", "Negative"])

if st.button("🔍 Lancer la veille"):
    newsdata = fetch_newsdata_articles("DS Automobiles", nb_articles)
    mediastack = fetch_mediastack_articles("DS Automobiles", nb_articles)
    rss = fetch_rss_articles("DS Automobiles")
    articles = pd.DataFrame(newsdata + mediastack + rss)

    if not articles.empty:
        articles["date"] = pd.to_datetime(articles["date"], errors="coerce")
        articles = articles[articles["date"].notna()]
        articles = articles[articles["date"] >= pd.Timestamp.now() - pd.Timedelta(days=nb_jours)]
        with st.spinner("Analyse en cours..."):
            articles[['résumé', 'ton', 'modèle']] = articles.apply(analyser_article, axis=1)

        if traduire:
            from transformers import pipeline as pipe
            traducteur = pipe("translation", model="Helsinki-NLP/opus-mt-mul-fr")
            articles["résumé"] = articles["résumé"].apply(lambda txt: traducteur(txt[:512])[0]["translation_text"] if txt and txt != "Aucun contenu" else txt)

        articles['date'] = pd.to_datetime(articles['date'], errors='coerce')
        articles = articles.sort_values(by="date", ascending=False).reset_index(drop=True)

        for _, row in articles.iterrows():
            envoyer_notif_slack(row)

        if filtre_modele != "Tous":
            articles = articles[articles['modèle'] == filtre_modele]
        if filtre_ton != "Tous":
            articles = articles[articles['ton'] == filtre_ton]

        mentions_today = len(articles)
        moyenne_7j = 25 / 7
        indice = int((mentions_today / max(moyenne_7j, 1)) * 50 + random.randint(0, 20))
        niveau = '🔴 Pic' if indice > 75 else '🟡 Stable' if indice > 50 else '🟢 Faible'

        st.metric("Indice de notoriété", f"{indice}/100", niveau)
        st.dataframe(articles[['date', 'titre', 'modèle', 'ton', 'résumé', 'source', 'lien']])
    else:
        st.warning("Aucun article trouvé.")

def fetch_rss_articles(query):
    import feedparser
    articles = []
    for feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries:
                title = entry.get("title", "")
                desc = entry.get("summary", "")
                if any(word.lower() in title.lower() or word.lower() in desc.lower() for word in ["DS Automobiles", "DS7", "DS N4", "DS N8", "Stellantis"]):
                    articles.append({
                        "date": entry.get("published", ""),
                        "titre": title,
                        "contenu": desc,
                        "source": feed_url,
                        "lien": entry.get("link", "")
                    })
        except:
            continue
    return articles
