### Complex dataset

# import os
# import sys
# import pandas as pd
# import numpy as np
# from pathlib import Path
# from dotenv import load_dotenv
# from ollama import Client as OllamaClient
# from pymilvus import MilvusClient

# # Giskard imports
# from giskard.rag import KnowledgeBase, generate_testset
# from giskard.llm.client import ChatMessage
# from giskard.rag.question_generators import SimpleQuestionsGenerator

# # =========================================================
# # 1. CONFIGURATION ET IMPORTS
# # =========================================================
# ROOT_DIR = Path(__file__).resolve().parent.parent.parent
# sys.path.append(str(ROOT_DIR))

# # Chargement du .env
# env_path = ROOT_DIR / ".env"
# load_dotenv(env_path)

# try:
#     from src.utils.custom_embedding import CustomEmbedder
#     print("✅ CustomEmbedder importé avec succès.")
# except ImportError as e:
#     print(f"Erreur d'import de CustomEmbedder : {e}")
#     sys.exit(1)

# # Variables d'environnement
# MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
# COLLECTION_NAME = os.getenv("MILVUS_COLLECTION")
# OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
# OLLAMA_MODEL = "llama3.1:8b" #os.getenv("OLLAMA_MODEL", "llama3.2:3b")
# HF_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME")

# # =========================================================
# # 2. WRAPPERS POUR COMPATIBILITÉ GISKARD (CORRIGÉS)
# # =========================================================

# class GiskardOllamaLLM:
#     def __init__(self, host, model):
#         self.client = OllamaClient(host=host)
#         self.model = model

#     # L'ajout de **kwargs est CRUCIAL pour absorber caller_id, seed, etc.
#     def complete(self, messages, temperature=0.0, max_tokens=None, **kwargs):
#         ollama_messages = [{"role": m.role, "content": m.content} for m in messages]
        
#         try:
#             resp = self.client.chat(
#                 model=self.model,
#                 messages=ollama_messages,
#                 options={
#                     "temperature": temperature, 
#                     "num_predict": max_tokens if max_tokens else 512
#                 }
#             )
#             # Gestion flexible du format de réponse Ollama
#             if hasattr(resp, 'message'):
#                 content = resp.message.content
#             else:
#                 content = resp['message']['content']
                
#             return ChatMessage(role="assistant", content=content)
#         except Exception as e:
#             print(f"Erreur Ollama : {e}")
#             return ChatMessage(role="assistant", content="Erreur de génération LLM.")

# class GiskardCustomEmbedder:
#     def __init__(self, model_name):
#         self.embedder = CustomEmbedder(model_name)

#     def embed(self, texts):
#         # Giskard attend un array numpy de vecteurs
#         embeddings = [self.embedder(t) for t in texts]
#         return np.array(embeddings)

# # =========================================================
# # 3. LOGIQUE PRINCIPALE
# # =========================================================

# def fetch_data_from_milvus(limit=150):
#     print(f"Connexion à Milvus ({MILVUS_URI})...")
#     client = MilvusClient(MILVUS_URI)
    
#     results = client.query(
#         collection_name=COLLECTION_NAME,
#         filter="", 
#         output_fields=["text", "source", "section_title"],
#         limit=limit
#     )
    
#     df = pd.DataFrame(results)
#     if df.empty:
#         raise ValueError(f"La collection {COLLECTION_NAME} est vide !")
        
#     print(f"✅ {len(df)} chunks récupérés.")
#     return df

# def run_testset_generation():
#     # 1. Extraction
#     df_raw = fetch_data_from_milvus(limit=100)

#     # 2. Formatage
#     df_kb = df_raw.copy()
#     df_kb["document"] = (
#         "Source: " + df_kb["source"].astype(str) + "\n"
#         "Section: " + df_kb.get("section_title", "").fillna("").astype(str) + "\n\n"
#         + df_kb["text"].astype(str)
#     )

#     # 3. Initialisation des composants Giskard
#     llm_client = GiskardOllamaLLM(OLLAMA_URL, OLLAMA_MODEL)
#     embedding_model = GiskardCustomEmbedder(HF_EMBEDDING_MODEL)

#     print("Initialisation de la KnowledgeBase...")
#     knowledge_base = KnowledgeBase.from_pandas(
#         df_kb,
#         columns=["document"],
#         llm_client=llm_client,
#         embedding_model=embedding_model,
#     )

#     print(f"Génération du testset via {OLLAMA_MODEL}...")
    
#     # Utilisation explicite du générateur simple pour éviter le clustering complexe
#     simple_gen = SimpleQuestionsGenerator(llm_client=llm_client)

#     testset = generate_testset(
#         knowledge_base,
#         num_questions=10, 
#         language="fr",
#         agent_description="Assistant technique pour le projet RAG",
#         question_generators=[simple_gen]
#     )

#     # 4. Sauvegarde finale
#     output_dir = Path("data")
#     output_dir.mkdir(exist_ok=True)
#     save_path = output_dir / "testset.jsonl"
    
#     # Vérification si des questions ont bien été produites
#     df_final = testset.to_pandas()
#     if len(df_final) > 0:
#         testset.save(str(save_path))
#         print(f"✅ SUCCÈS ! Fichier généré : {save_path}")
#         print(f"Aperçu des questions :\n{df_final['question'].head()}")
#     else:
#         print("ÉCHEC : Aucune question n'a été générée. Vérifiez les logs Ollama.")

# if __name__ == "__main__":
#     try:
#         run_testset_generation()
#     except Exception as e:
#         print(f"Erreur fatale : {e}")
#         import traceback
#         traceback.print_exc()



# ### Simple dataset
# import os
# import sys
# import pandas as pd
# import numpy as np
# from pathlib import Path
# from dotenv import load_dotenv
# from ollama import Client as OllamaClient
# from pymilvus import MilvusClient

# # Giskard imports
# from giskard.rag import KnowledgeBase, generate_testset
# from giskard.llm.client import ChatMessage
# from giskard.rag.question_generators import SimpleQuestionsGenerator

# # =========================================================
# # 1. CONFIGURATION ET IMPORTS
# # =========================================================
# ROOT_DIR = Path(__file__).resolve().parent.parent.parent
# sys.path.append(str(ROOT_DIR))

# # Chargement du .env
# env_path = ROOT_DIR / ".env"
# load_dotenv(env_path)

# try:
#     from src.utils.custom_embedding import CustomEmbedder
#     print("CustomEmbedder importé avec succès.")
# except ImportError as e:
#     print(f"Erreur d'import de CustomEmbedder : {e}")
#     sys.exit(1)

# # Variables d'environnement
# MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
# COLLECTION_NAME = os.getenv("MILVUS_COLLECTION")
# OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
# OLLAMA_MODEL = "llama3.1:8b" #os.getenv("OLLAMA_MODEL", "llama3.2:3b")
# HF_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME")

# # =========================================================
# # 2. WRAPPERS POUR COMPATIBILITÉ GISKARD (CORRIGÉS)
# # =========================================================

# class GiskardOllamaLLM:
#     def __init__(self, host, model):
#         self.client = OllamaClient(host=host)
#         self.model = model

#     # L'ajout de **kwargs est CRUCIAL pour absorber caller_id, seed, etc.
#     def complete(self, messages, temperature=0.0, max_tokens=None, **kwargs):
#         ollama_messages = [{"role": m.role, "content": m.content} for m in messages]
        
#         try:
#             resp = self.client.chat(
#                 model=self.model,
#                 messages=ollama_messages,
#                 options={
#                     "temperature": temperature, 
#                     "num_predict": max_tokens if max_tokens else 512
#                 }
#             )
#             # Gestion flexible du format de réponse Ollama
#             if hasattr(resp, 'message'):
#                 content = resp.message.content
#             else:
#                 content = resp['message']['content']
                
#             return ChatMessage(role="assistant", content=content)
#         except Exception as e:
#             print(f"⚠️ Erreur Ollama : {e}")
#             return ChatMessage(role="assistant", content="Erreur de génération LLM.")

# class GiskardCustomEmbedder:
#     def __init__(self, model_name):
#         self.embedder = CustomEmbedder(model_name)

#     def embed(self, texts):
#         # Giskard attend un array numpy de vecteurs
#         embeddings = [self.embedder(t) for t in texts]
#         return np.array(embeddings)

# # =========================================================
# # 3. LOGIQUE PRINCIPALE
# # =========================================================

# def fetch_data_from_milvus(limit=150):
#     print(f"Connexion à Milvus ({MILVUS_URI})...")
#     client = MilvusClient(MILVUS_URI)
    
#     results = client.query(
#         collection_name=COLLECTION_NAME,
#         filter="", 
#         output_fields=["text", "source", "section_title"],
#         limit=limit
#     )
    
#     df = pd.DataFrame(results)
#     if df.empty:
#         raise ValueError(f"La collection {COLLECTION_NAME} est vide !")
        
#     print(f"✅ {len(df)} chunks récupérés.")
#     return df

# def run_testset_generation():
#     # 1. Extraction
#     df_raw = fetch_data_from_milvus(limit=100)

#     # 2. Formatage
#     df_kb = df_raw.copy()
#     df_kb["document"] = (
#         "Source: " + df_kb["source"].astype(str) + "\n"
#         "Section: " + df_kb.get("section_title", "").fillna("").astype(str) + "\n\n"
#         + df_kb["text"].astype(str)
#     )

#     # 3. Initialisation des composants Giskard
#     llm_client = GiskardOllamaLLM(OLLAMA_URL, OLLAMA_MODEL)
#     embedding_model = GiskardCustomEmbedder(HF_EMBEDDING_MODEL)

#     print("Initialisation de la KnowledgeBase...")
#     knowledge_base = KnowledgeBase.from_pandas(
#         df_kb,
#         columns=["document"],
#         llm_client=llm_client,
#         embedding_model=embedding_model,
#     )

#     print(f"Génération du testset via {OLLAMA_MODEL}...")
    
#     # Utilisation explicite du générateur simple pour éviter le clustering complexe
#     simple_gen = SimpleQuestionsGenerator(llm_client=llm_client)

#     testset = generate_testset(
#         knowledge_base,
#         num_questions=10, 
#         language="fr",
#         agent_description="Assistant technique pour le projet RAG",
#         question_generators=[simple_gen]
#     )

#     # 4. Sauvegarde finale
#     output_dir = Path("data")
#     output_dir.mkdir(exist_ok=True)
#     save_path = output_dir / "testset.jsonl"
    
#     # Vérification si des questions ont bien été produites
#     df_final = testset.to_pandas()
#     if len(df_final) > 0:
#         testset.save(str(save_path))
#         print(f"✅ SUCCÈS ! Fichier généré : {save_path}")
#         print(f"Aperçu des questions :\n{df_final['question'].head()}")
#     else:
#         print("ÉCHEC : Aucune question n'a été générée. Vérifiez les logs Ollama.")

# if __name__ == "__main__":
#     try:
#         run_testset_generation()
#     except Exception as e:
#         print(f"Erreur fatale : {e}")
#         import traceback
#         traceback.print_exc()


import os
import sys
import time
import pandas as pd
import numpy as np
import mlflow
from pathlib import Path
from dotenv import load_dotenv
from ollama import Client as OllamaClient
from pymilvus import MilvusClient

# Giskard imports
from giskard.rag import KnowledgeBase, generate_testset
from giskard.llm.client import ChatMessage
from giskard.rag.question_generators import SimpleQuestionsGenerator

# =========================================================
# 1. CONFIGURATION ET IMPORTS
# =========================================================
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

load_dotenv(ROOT_DIR / ".env")

try:
    from src.utils.custom_embedding import CustomEmbedder
    print("✅ CustomEmbedder importé avec succès.")
except ImportError as e:
    print(f"Erreur d'import de CustomEmbedder : {e}")
    sys.exit(1)

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = "llama3.1:8b" 
HF_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME")

# =========================================================
# 2. WRAPPERS GISKARD
# =========================================================

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

# =========================================================
# 3. LOGIQUE PRINCIPALE
# =========================================================

def fetch_data_from_milvus(limit=150):
    client = MilvusClient(MILVUS_URI)
    results = client.query(
        collection_name=COLLECTION_NAME,
        filter="", 
        output_fields=["text", "source", "section_title"],
        limit=limit
    )
    df = pd.DataFrame(results)
    if df.empty: raise ValueError("Collection vide !")
    return df

def run_testset_generation():
    mlflow.set_experiment("Giskard_RAG_Testing")
    
    with mlflow.start_run(run_name=f"Gen_{OLLAMA_MODEL}_{time.strftime('%H%M%S')}"):
        # --- PARAMÈTRES ET TAGS ---
        mlflow.log_params({
            "llm_model": OLLAMA_MODEL,
            "embedding_model": HF_EMBEDDING_MODEL,
            "milvus_collection": COLLECTION_NAME,
            "num_questions_requested": 10
        })
        mlflow.set_tags({
            "project": "RAG_Internal",
            "language": "fr",
            "generator_type": "SimpleQuestions"
        })

        # 1. Extraction et Stats KB
        df_raw = fetch_data_from_milvus(limit=100)
        mlflow.log_metric("kb_total_chunks", len(df_raw))
        mlflow.log_metric("kb_avg_chunk_size", df_raw['text'].str.len().mean())

        # 2. Formatage
        df_kb = df_raw.copy()
        df_kb["document"] = "Source: " + df_kb["source"].astype(str) + "\n\n" + df_kb["text"].astype(str)

        # 3. Initialisation Giskard
        llm_client = GiskardOllamaLLM(OLLAMA_URL, OLLAMA_MODEL)
        embedding_model = GiskardCustomEmbedder(HF_EMBEDDING_MODEL)

        knowledge_base = KnowledgeBase.from_pandas(
            df_kb, columns=["document"], 
            llm_client=llm_client, embedding_model=embedding_model
        )

        # 4. Génération avec mesure du temps
        print(f"Génération en cours via {OLLAMA_MODEL}...")
        start_time = time.time()
        
        simple_gen = SimpleQuestionsGenerator(llm_client=llm_client)
        testset = generate_testset(
            knowledge_base,
            num_questions=10, 
            language="fr",
            agent_description="Assistant technique RAG",
            question_generators=[simple_gen]
        )
        
        duration = time.time() - start_time
        df_final = testset.to_pandas()

        # 5. Métriques de sortie et Artifacts
        mlflow.log_metric("gen_duration_sec", duration)
        mlflow.log_metric("actual_questions_count", len(df_final))
        
        if len(df_final) > 0:
            mlflow.log_metric("sec_per_question", duration / len(df_final))
            
            # Diversité des questions (si présentes dans les métadonnées)
            if 'metadata' in df_final.columns:
                q_types = df_final['metadata'].apply(lambda x: x.get('question_type', 'unknown')).value_counts()
                for q_type, count in q_types.items():
                    mlflow.log_metric(f"type_{q_type.replace(' ', '_')}", count)

            # Sauvegardes
            output_dir = Path("data")
            output_dir.mkdir(exist_ok=True)
            save_path = output_dir / "testset.jsonl"
            testset.save(str(save_path))
            
            mlflow.log_artifact(str(save_path.absolute()), artifact_path="datasets")
            mlflow.log_table(data=df_final, artifact_file="testset_preview.json")
            
            print(f"✅ Terminé en {duration:.2f}s. Logs dispos dans MLflow (Model Training).")
        else:
            mlflow.set_tag("status", "failed_no_questions")
            print("ÉCHEC : Aucune question générée.")

if __name__ == "__main__":
    try:
        run_testset_generation()
    except Exception as e:
        mlflow.log_param("error_msg", str(e))
        print(f"Erreur : {e}")
        import traceback
        traceback.print_exc()