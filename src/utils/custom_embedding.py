# faire en ligne de commande : huggingface-cli login
from sentence_transformers import SentenceTransformer
import torch
import os

from dotenv import load_dotenv

load_dotenv()

class CustomEmbedder:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base", device: str = None):
        
        hf_token = os.getenv("HF_TOKEN") 

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Chargement du modèle {model_name} sur {self.device}...")
        self.model = SentenceTransformer(
            model_name, 
            device=self.device,
            use_auth_token=hf_token
        )

    def __call__(self, text: str) -> list[float]:
        """
        Cette méthode rend l'instance "callable" (appelable comme une fonction).
        Elle correspond au 'emb_fn' attendu par votre mineur.
        """
        # On encode le texte. convert_to_numpy=True par défaut.
        # .tolist() transforme le vecteur numpy en liste de floats pour Milvus.
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()