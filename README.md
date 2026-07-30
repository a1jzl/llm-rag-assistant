# LLM RAG Assistant

Assistant conversationnel base sur une architecture RAG (Retrieval-Augmented Generation), capable de repondre a des questions sur un corpus de documents metier (contrats, documentation interne, articles reglementaires, etc.) en combinant recherche semantique et generation de texte par un LLM.

## 1. Contexte et objectif

Les modeles de langage generalistes hallucinent facilement lorsqu'on leur pose des questions pointues sur un domaine specifique qu'ils n'ont pas vu pendant leur entrainement. Ce projet propose une solution concrete a ce probleme en ancrant les reponses du LLM dans un corpus documentaire propre, verifiable et mis a jour independamment du modele.

Cas d'usage cible : assistant de support interne capable de repondre a des questions sur une base de connaissances (procedures internes, documentation technique, FAQ produit).

## 2. Architecture

Le pipeline se decompose en quatre etapes :

1. Ingestion : chargement des documents (PDF, Markdown, HTML), decoupage en chunks avec chevauchement (chunking semantique).
2. Indexation : calcul des embeddings de chaque chunk (sentence-transformers ou API d'embeddings) et stockage dans une base vectorielle (FAISS en local, Chroma en alternative).
3. Recherche (retrieval) : au moment de la question, recherche des k chunks les plus proches semantiquement de la requete utilisateur.
4. Generation : construction d'un prompt combinant la question et les chunks recuperes, puis appel au LLM pour generer une reponse sourcee.

## 3. Stack technique

- Python 3.11
- LangChain pour l'orchestration du pipeline RAG
- FAISS pour l'indexation vectorielle
- sentence-transformers pour les embeddings
- API LLM (OpenAI ou modele open source via Ollama) pour la generation
- Streamlit pour l'interface de demonstration
- RAGAS pour l'evaluation de la qualite des reponses (pertinence, fidelite au contexte)

## 4. Structure du repo

```
llm-rag-assistant/
  data/                 documents sources (non versionnes)
  src/
    ingest.py           chargement et chunking des documents
    build_index.py      calcul des embeddings et creation de l'index FAISS
    retriever.py        recherche semantique dans l'index
    generate.py         construction du prompt et appel au LLM
    app.py              interface Streamlit
  evaluation/
    eval_ragas.py       evaluation automatique des reponses
  requirements.txt
  .github/workflows/ci.yml
```

## 5. Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Definir une variable d'environnement OPENAI_API_KEY (ou configurer Ollama en local) avant de lancer l'application.

## 6. Utilisation

```bash
python src/ingest.py --input data/
python src/build_index.py
streamlit run src/app.py
```

## 7. Evaluation

Le dossier evaluation/ contient un script qui mesure, sur un jeu de questions-reponses annote manuellement, la pertinence du contexte recupere et la fidelite de la reponse generee par rapport aux sources.

## 8. Limites et pistes d'amelioration

- Le decoupage en chunks fixes peut casser le sens de certains passages : une strategie de chunking par section semantique serait plus robuste.
- Pas encore de gestion du re-ranking des documents recuperes (cross-encoder) pour ameliorer la precision.
- Le monitoring en production (derive du corpus, feedback utilisateur) reste a implementer.

## 9. Auteur

Projet realise dans le cadre d'une recherche d'alternance en data/IA.
