import os
import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from ollama import Client as OllamaClient
from pymilvus import MilvusClient
from langfuse import Langfuse

# Giskard imports
from giskard.rag import KnowledgeBase, generate_testset
from giskard.llm.client import ChatMessage
from giskard.rag.question_generators import DoubleQuestionsGenerator, SituationalQuestionsGenerator

# =========================================================
# 1. CONFIGURATION
# =========================================================
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))
load_dotenv(ROOT_DIR / ".env")

try:
    from src.utils.custom_embedding import CustomEmbedder
    print("✅ CustomEmbedder importé.")
except ImportError:
    print("Erreur import CustomEmbedder")
    sys.exit(1)

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = "llama3.1:8b" 
HF_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME")



class GiskardOllamaLLM:
    def __init__(self, host, model):
        self.client = OllamaClient(host=host)
        self.model = model

    def complete(self, messages, temperature=0.0, max_tokens=None, **kwargs):
        ollama_messages = [{"role": m.role, "content": m.content} for m in messages]
        try:
            resp = self.client.chat(
                model=self.model,
                messages=ollama_messages,
                options={"temperature": temperature, "num_predict": max_tokens or 512}
            )
            content = resp.message.content if hasattr(resp, 'message') else resp['message']['content']
            return ChatMessage(role="assistant", content=content)
        except Exception as e:
            return ChatMessage(role="assistant", content=f"Erreur LLM: {e}")

class GiskardCustomEmbedder:
    def __init__(self, model_name):
        self.embedder = CustomEmbedder(model_name)
    def embed(self, texts):
        return np.array([self.embedder(t) for t in texts])

def is_question_relevant(llm_client, question, context):
    """ 
    JUGE SÉVÈRE : Évalue si la question est auto-suffisante et technique.
    Rejette les questions sans sujet ou citant 'le texte'.
    """
    prompt = [
        ChatMessage(role="system", content="""Tu es un contrôleur qualité RAG expert. 
        Ton but est de REJETER les questions trop vagues ou orphelines."""),
        ChatMessage(role="user", content=(
            f"Contexte: {context}\n\n"
            f"Question: {question}\n\n"
            "Réponds UNIQUEMENT par OUI ou NON selon ces critères :\n"
            "1. La question contient-elle le nom du sujet/modèle (ex: DeepSeek, YARN, etc.) ?\n"
            "2. La question évite-t-elle de dire 'le document' ou 'ce texte' ?\n"
            "3. La question est-elle assez précise pour qu'un expert comprenne de quoi on parle sans lire le contexte ?"
        ))
    ]
    response = llm_client.complete(prompt, temperature=0.0).content.strip().upper()
    return "OUI" in response

def upload_to_langfuse(df, dataset_name):
    langfuse = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "http://localhost:3000")
    )

    print(f"Upload vers Langfuse : {dataset_name}...")
    langfuse.create_dataset(name=dataset_name)

    for _, row in df.iterrows():
        # Extraction propre des métadonnées giskard
        meta = row.get("metadata", {})
        source_val = meta[0].get("source", "N/A") if isinstance(meta, list) and len(meta) > 0 else "N/A"

        langfuse.create_dataset_item(
            dataset_name=dataset_name,
            input=row["question"],
            expected_output=row["reference_answer"],
            metadata={
                "source": source_val,
                "reference_context": row["reference_context"], # Vital pour Ragas
                "model_generator": OLLAMA_MODEL,
                "generation_type": "Giskard_Complex"
            }
        )
    print("✅ Dataset synchronisé avec Langfuse.")


def fetch_data_from_milvus(limit=100):
    client = MilvusClient(MILVUS_URI)
    results = client.query(collection_name=COLLECTION_NAME, filter="", 
                           output_fields=["text", "source"], limit=limit)
    return pd.DataFrame(results)

def run_testset_generation():
    target_questions = 10
    valid_rows = []
    attempts = 0
    max_attempts = 3 
    
    df_raw = fetch_data_from_milvus()
    df_kb = df_raw.copy()
    df_kb["document"] = "Source: " + df_kb["source"].astype(str) + "\n\n" + df_kb["text"].astype(str)

    llm_client = GiskardOllamaLLM(OLLAMA_URL, OLLAMA_MODEL)
    embedding_model = GiskardCustomEmbedder(HF_EMBEDDING_MODEL)

    kb = KnowledgeBase.from_pandas(df_kb, columns=["document"], 
                                  llm_client=llm_client, embedding_model=embedding_model)

    # Prompt Expert pour forcer l'inclusion du sujet
    AGENT_DESC = """Tu es un ingénieur en IA senior. 
    Chaque question que tu poses DOIT être auto-suffisante et inclure le nom du modèle ou du concept 
    spécifique mentionné dans le texte (ex: 'DeepSeek-V2', 'MoE', 'YARN'). 
    NE JAMAIS dire 'le modèle' ou 'ce document' sans préciser le nom."""

    generators = [
        DoubleQuestionsGenerator(llm_client=llm_client),
        SituationalQuestionsGenerator(llm_client=llm_client)
    ]

    print(f"Génération Haute Qualité (Cible: {target_questions})")
    start_time = time.time()

    while len(valid_rows) < target_questions and attempts < max_attempts:
        attempts += 1
        needed = (target_questions - len(valid_rows)) * 2 # On demande plus pour filtrer
        
        batch_testset = generate_testset(
            kb,
            num_questions=needed,
            language="fr",
            agent_description=AGENT_DESC,
            question_generators=generators
        )
        
        batch_df = batch_testset.to_pandas()
        
        for _, row in batch_df.iterrows():
            if len(valid_rows) >= target_questions:
                break
            
            
            if is_question_relevant(llm_client, row['question'], row['reference_context']):
                valid_rows.append(row)
                print(f" ✅ Validée: {row['question'][:70]}...")
            else:
                print(f" Rejetée (Sujet manquant/trop vague): {row['question'][:70]}...")

    df_final = pd.DataFrame(valid_rows)
    if not df_final.empty:
        dataset_name = f"Giskard_Expert_V2_{time.strftime('%Y%m%d_%H%M')}"
        upload_to_langfuse(df_final, dataset_name)
        print(f"\nTerminé ! {len(df_final)} questions expertes sauvegardées.")
    else:
        print("Échec : Aucune question n'a passé le filtre.")

if __name__ == "__main__":
    run_testset_generation()