# # faire en ligne de commande : huggingface-cli login
# from sentence_transformers import SentenceTransformer
# import torch
# import os

# from dotenv import load_dotenv

# load_dotenv()

# class CustomEmbedder:
#     def __init__(self, model_name: str = "intfloat/multilingual-e5-base", device: str = None):
        
#         hf_token = os.getenv("HF_TOKEN") 

#         if device is None:
#             self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         else:
#             self.device = device
            
#         print(f"Chargement du modèle {model_name} sur {self.device}...")
#         self.model = SentenceTransformer(
#             model_name, 
#             device=self.device,
#             use_auth_token=hf_token
#         )

#     def __call__(self, text: str) -> list[float]:
#         """
#         Cette méthode rend l'instance "callable" (appelable comme une fonction).
#         Elle correspond au 'emb_fn' attendu par votre mineur.
#         """
#         # On encode le texte. convert_to_numpy=True par défaut.
#         # .tolist() transforme le vecteur numpy en liste de floats pour Milvus.
#         embedding = self.model.encode(text, normalize_embeddings=True)
#         return embedding.tolist()





import requests
import asyncio
import torch
from dotenv import load_dotenv

load_dotenv()

import os
import asyncio
import httpx

class CustomEmbedder:
    def __init__(self, model_name: str = None):
        self.tei_url = os.getenv("TEI_ENDPOINT")
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-base")
        self._client = None
        self._local_model = None  # Cache pour le modèle local

    @property
    def client(self):
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(20.0, connect=2.0),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
            )
        return self._client

    async def _get_local_model(self):
        """Charge le modèle en RAM une seule fois si TEI n'est pas dispo."""
        if self._local_model is None:
            from sentence_transformers import SentenceTransformer
            # On détecte le device (CPU/MPS pour Mac/CUDA)
            import torch
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
            print(f"Chargement du modèle local sur {device}...")
            self._local_model = SentenceTransformer(self.model_name, device=device)
        return self._local_model

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts: return []
        
        # PRIORITÉ : Mode TEI (Ultra rapide)
        if self.tei_url:
            try:
                response = await self.client.post(
                    f"{self.tei_url}/embed", 
                    json={"inputs": texts}
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                print(f"Erreur TEI, tentative de bascule locale: {e}")
        
        # SECOURS : Mode Local (Chargé une seule fois)
        model = await self._get_local_model()
        # On utilise to_thread car model.encode est synchrone et bloque le CPU
        embeddings = await asyncio.to_thread(model.encode, texts, normalize_embeddings=True)
        return embeddings.tolist()

    async def embed_query(self, text: str) -> list[float]:
        res = await self.embed_documents([text])
        return res[0]

    async def __call__(self, text: str) -> list[float]:
        return await self.embed_query(text)

    async def close(self):
        """À appeler lors de la fermeture de l'app (shutdown event)"""
        if self._client:
            await self._client.aclose()

# class CustomEmbedder:
#     def __init__(self, model_name: str = "intfloat/multilingual-e5-base", device: str = None):
#         # On récupère l'URL de TEI depuis le .env (ex: http://localhost:8083)
#         self.tei_url = os.getenv("TEI_ENDPOINT")
        
#         if self.tei_url:
#             print(f"Mode TEI activé : Utilisation du serveur à l'adresse {self.tei_url}")
#             # En mode TEI, on n'a pas besoin de charger le modèle en local
#             self.model = None 
#         else:
#             # Comportement d'origine si TEI n'est pas configuré
#             hf_token = os.getenv("HF_TOKEN")
#             if device is None:
#                 self.device = "cuda" if torch.cuda.is_available() else "cpu"
#             else:
#                 self.device = device
                
#             print(f"Mode Local : Chargement du modèle {model_name} sur {self.device}...")
#             self.model = SentenceTransformer(
#                 model_name, 
#                 device=self.device,
#                 token=hf_token
#             )

#     def __call__(self, text: str) -> list[float]:
#         # Si TEI est configuré, on fait une requête HTTP
#         if self.tei_url:
#             try:
#                 response = requests.post(
#                     f"{self.tei_url}/embed",
#                     json={"inputs": text},
#                     timeout=10
#                 )
#                 response.raise_for_status()
#                 # TEI renvoie une liste de listes (car il peut gérer des batchs)
#                 # On prend le premier élément
#                 return response.json()[0]
#             except Exception as e:
#                 print(f"Erreur de connexion à TEI: {e}. Vérifiez que le conteneur tourne sur {self.tei_url}")
#                 raise e
        
#         # Sinon, on utilise le modèle local (comportement d'origine)
#         embedding = self.model.encode(text, normalize_embeddings=True)
#         return embedding.tolist()