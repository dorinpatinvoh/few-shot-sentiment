import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict
import copy

class TemporaryParameters:
    def __init__(self, model, temporary_weights):
        self.model = model
        self.temporary_weights = temporary_weights
        self.original_weights = None

    def __enter__(self):
        # Sauvegarder les poids originaux
        self.original_weights = [p.data.clone() for p in self.model.parameters()]
        
        # Appliquer les poids temporaires
        for p, w in zip(self.model.parameters(), self.temporary_weights):
            p.data.copy_(w.data)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restaurer les poids originaux
        for p, w in zip(self.model.parameters(), self.original_weights):
            p.data.copy_(w.data)

class FewShotLearner(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased", num_classes: int = 2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_classes)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def forward(self, support_set: Dict[str, torch.Tensor], query: Dict[str, torch.Tensor]):
        # S'assurer que seuls les arguments valides sont passés à l'encodeur
        encoder_inputs_support = {
            k: v for k, v in support_set.items() 
            if k in ['input_ids', 'attention_mask']
        }
        encoder_inputs_query = {
            k: v for k, v in query.items() 
            if k in ['input_ids', 'attention_mask']
        }
        
        # Encoder le support set (exemples d'apprentissage)
        support_embeddings = self.encoder(**encoder_inputs_support).last_hidden_state[:, 0, :]
        
        # Encoder la requête
        query_embeddings = self.encoder(**encoder_inputs_query).last_hidden_state[:, 0, :]
        
        # Calculer la similarité cosinus entre la requête et le support set
        similarities = torch.cosine_similarity(query_embeddings.unsqueeze(1), 
                                            support_embeddings.unsqueeze(0), dim=2)
        
        return self.classifier(query_embeddings), similarities

def prepare_few_shot_batch(texts: List[str], labels: List[int], tokenizer):
    """Prépare un lot pour l'apprentissage few-shot"""
    if not texts:  # Vérification si la liste est vide
        raise ValueError("La liste de textes ne peut pas être vide")
        
    encodings = tokenizer(
        texts, 
        padding=True, 
        truncation=True, 
        return_tensors="pt",
        max_length=512
    )
    return {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": torch.tensor(labels)
        # Suppression de "texts" car ce n'est pas un tensor
    }

class SelfSupervisedTask(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.mlm_head = nn.Linear(hidden_size, 30522)  # Taille du vocabulaire BERT
        self.nsp_head = nn.Linear(hidden_size, 2)

    def forward(self, embeddings):
        mlm_output = self.mlm_head(embeddings)
        nsp_output = self.nsp_head(embeddings[:, 0, :])  # Utilise le token [CLS]
        return mlm_output, nsp_output

class MAMLFewShotLearner(FewShotLearner):
    def __init__(self, model_name: str = "bert-base-uncased", num_classes: int = 2, alpha: float = 0.01):
        super().__init__(model_name, num_classes)
        self.self_supervised = SelfSupervisedTask(self.encoder.config.hidden_size)
        self.alpha = alpha  # Taux d'apprentissage interne de MAML
        
    def self_supervised_forward(self, texts: List[str]):
        # Masquer aléatoirement 15% des tokens
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        masked_inputs = self._mask_tokens(inputs["input_ids"])
        
        embeddings = self.encoder(
            input_ids=masked_inputs,
            attention_mask=inputs["attention_mask"]
        ).last_hidden_state
        
        return self.self_supervised(embeddings)

    def _mask_tokens(self, inputs: torch.Tensor, masking_prob: float = 0.15):
        """Masque aléatoirement des tokens pour MLM"""
        mask = torch.rand(inputs.shape) < masking_prob
        masked_inputs = inputs.clone()
        masked_inputs[mask] = self.tokenizer.mask_token_id
        return masked_inputs
    
    def maml_inner_loop(self, support_set: Dict[str, torch.Tensor], num_steps: int = 5):
        """Boucle interne MAML pour adaptation rapide"""
        # Créer un dictionnaire des paramètres
        fast_weights = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                fast_weights[name] = param.clone().detach().requires_grad_(True)
        
        for _ in range(num_steps):
            # Forward pass avec les poids actuels
            encoder_inputs = {
                k: v for k, v in support_set.items() 
                if k in ['input_ids', 'attention_mask']
            }
            
            # Calculer la sortie avec les poids temporaires
            with torch.enable_grad():
                # Sauvegarder les poids originaux
                original_weights = {}
                for name, param in self.named_parameters():
                    if param.requires_grad:
                        original_weights[name] = param.data.clone()
                        param.data = fast_weights[name].data
                
                # Forward pass
                logits, _ = self.forward(encoder_inputs, encoder_inputs)
                loss = nn.CrossEntropyLoss()(logits, support_set["labels"])
                
                # Calculer les gradients
                grads = torch.autograd.grad(loss, [fast_weights[name] for name, param in self.named_parameters() if param.requires_grad], 
                                          create_graph=True, allow_unused=True)
                
                # Mettre à jour les poids rapides
                for (name, param), grad in zip([(name, param) for name, param in self.named_parameters() if param.requires_grad], grads):
                    if grad is not None:
                        fast_weights[name] = fast_weights[name] - self.alpha * grad
                
                # Restaurer les poids originaux
                for name, param in self.named_parameters():
                    if param.requires_grad:
                        param.data = original_weights[name]
        
        return list(fast_weights.values())

def train_step(model: MAMLFewShotLearner, optimizer, support_set, query_set):
    # Déplacer les tenseurs sur le même device que le modèle
    device = next(model.parameters()).device
    
    # S'assurer que les tenseurs sont sur le bon device
    support_set = {k: v.to(device) for k, v in support_set.items()}
    query_set = {k: v.to(device) for k, v in query_set.items()}
    
    # Partie self-supervised
    encoder_inputs = {
        k: v for k, v in support_set.items() 
        if k in ['input_ids', 'attention_mask']
    }
    
    with torch.set_grad_enabled(True):
        # Self-supervised learning
        decoded_texts = model.tokenizer.batch_decode(encoder_inputs["input_ids"])
        mlm_output, nsp_output = model.self_supervised_forward(decoded_texts)
        
        batch_size = mlm_output.size(0)
        vocab_size = mlm_output.size(-1)
        mlm_labels = torch.randint(0, vocab_size, (batch_size, mlm_output.size(1)), device=device)
        nsp_labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        ssl_loss = calculate_ssl_loss(
            mlm_output, 
            nsp_output, 
            {"mlm_labels": mlm_labels, "nsp_labels": nsp_labels}
        )
        
        # MAML adaptation
        adapted_weights = model.maml_inner_loop(support_set)
        
        # Task-specific learning
        with TemporaryParameters(model, adapted_weights):
            query_logits, _ = model(
                {k: v for k, v in query_set.items() if k in ['input_ids', 'attention_mask']},
                {k: v for k, v in query_set.items() if k in ['input_ids', 'attention_mask']}
            )
            task_loss = nn.CrossEntropyLoss()(query_logits, query_set["labels"])
        
        # Total loss
        total_loss = task_loss + 0.1 * ssl_loss
        
        # Optimization step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    
    return total_loss.item()

def calculate_ssl_loss(mlm_output, nsp_output, batch):
    """Calcule la perte pour les tâches auto-supervisées"""
    device = mlm_output.device
    mlm_labels = batch["mlm_labels"].to(device)
    nsp_labels = batch["nsp_labels"].to(device)
    
    mlm_loss = nn.CrossEntropyLoss()(
        mlm_output.view(-1, mlm_output.size(-1)),
        mlm_labels.view(-1)
    )
    nsp_loss = nn.CrossEntropyLoss()(nsp_output, nsp_labels)
    return mlm_loss + nsp_loss

class DatasetManager:
    def __init__(self, train_path: str, val_path: str, test_path: str):
        # Données d'exemple directement dans le code pour commencer
        self.train_data = [
            {"text": "Excellent produit, je recommande", "label": 1},
            {"text": "Qualité médiocre, à éviter", "label": 0},
            {"text": "Super service, très satisfait", "label": 1},
            {"text": "Délai de livraison trop long", "label": 0},
            {"text": "Parfait, rien à redire", "label": 1},
            {"text": "Produit défectueux", "label": 0},
            {"text": "Service client excellent", "label": 1},
            {"text": "Ne fonctionne pas", "label": 0}
        ]
        self.val_data = self.train_data[:4]  # Utiliser une partie des données pour la validation
        self.test_data = self.train_data[4:]  # Et une autre pour le test
    
    def create_episode(self, data: List[Dict], n_support: int = 2, n_query: int = 2):
        """Crée un épisode d'apprentissage few-shot avec des valeurs plus petites"""
        if not data:
            raise ValueError("Les données ne peuvent pas être vides")
            
        classes = list(set(d["label"] for d in data))
        support_set = []
        query_set = []
        
        for c in classes:
            class_examples = [d for d in data if d["label"] == c]
            if len(class_examples) < (n_support + n_query):
                n_support = min(n_support, len(class_examples) // 2)
                n_query = min(n_query, len(class_examples) - n_support)
            
            support = class_examples[:n_support]
            query = class_examples[n_support:n_support + n_query]
            
            support_set.extend(support)
            query_set.extend(query)
            
        return support_set, query_set

def train_model(
    model: MAMLFewShotLearner,
    dataset_manager: DatasetManager,
    n_epochs: int = 100,
    n_episodes: int = 10,
    n_support: int = 5,
    n_query: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    best_val_loss = float('inf')
    best_model = None
    
    for epoch in range(n_epochs):
        model.train()
        train_losses = []
        
        # Boucle d'entraînement
        for _ in range(n_episodes):
            support_set, query_set = dataset_manager.create_episode(
                dataset_manager.train_data, n_support, n_query
            )
            
            support_batch = prepare_few_shot_batch(
                [x["text"] for x in support_set],
                [x["label"] for x in support_set],
                model.tokenizer
            )
            query_batch = prepare_few_shot_batch(
                [x["text"] for x in query_set],
                [x["label"] for x in query_set],
                model.tokenizer
            )
            
            # Déplacer les tenseurs sur le device
            support_batch = {k: v.to(device) for k, v in support_batch.items()}
            query_batch = {k: v.to(device) for k, v in query_batch.items()}
            
            loss = train_step(model, optimizer, support_batch, query_batch)
            train_losses.append(loss)
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for _ in range(n_episodes // 2):
                support_set, query_set = dataset_manager.create_episode(
                    dataset_manager.val_data, n_support, n_query
                )
                
                support_batch = prepare_few_shot_batch(
                    [x["text"] for x in support_set],
                    [x["label"] for x in support_set],
                    model.tokenizer
                )
                query_batch = prepare_few_shot_batch(
                    [x["text"] for x in query_set],
                    [x["label"] for x in query_set],
                    model.tokenizer
                )
                
                support_batch = {k: v.to(device) for k, v in support_batch.items()}
                query_batch = {k: v.to(device) for k, v in query_batch.items()}
                
                adapted_weights = model.maml_inner_loop(support_batch)
                with TemporaryParameters(model, adapted_weights):
                    logits, _ = model(query_batch, query_batch)
                    loss = nn.CrossEntropyLoss()(logits, query_batch["labels"])
                    val_losses.append(loss.item())
        
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_val_loss = sum(val_losses) / len(val_losses)
        
        # Mise à jour du scheduler
        scheduler.step(avg_val_loss)
        
        # Sauvegarde du meilleur modèle
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = copy.deepcopy(model.state_dict())
            torch.save(best_model, "best_model.pth")
        
        print(f"Epoch {epoch + 1}/{n_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print("-" * 50)
    
    return best_model

def main():
    # Initialisation avec moins d'époques et d'épisodes pour tester
    dataset_manager = DatasetManager(
        train_path="train.json",
        val_path="val.json",
        test_path="test.json"
    )
    
    model = MAMLFewShotLearner(model_name="camembert-base")
    
    # Entraînement avec des paramètres plus petits
    best_model = train_model(
        model=model,
        dataset_manager=dataset_manager,
        n_epochs=5,  # Réduit pour tester
        n_episodes=2,  # Réduit pour tester
        n_support=2,  # Réduit pour tester
        n_query=2  # Réduit pour tester
    )
    
    # Chargement du meilleur modèle pour l'évaluation
    model.load_state_dict(best_model)
    
    # Test sur quelques exemples
    test_texts = ["Ce produit est fantastique!", "Service client déplorable"]
    test_labels = [1, 0]
    
    test_batch = prepare_few_shot_batch(test_texts, test_labels, model.tokenizer)
    with torch.no_grad():
        logits, similarities = model(test_batch, test_batch)
        predictions = torch.argmax(logits, dim=1)
        
    print("Résultats des tests:")
    for text, pred in zip(test_texts, predictions):
        print(f"Texte: {text}")
        print(f"Prédiction: {'positif' if pred == 1 else 'négatif'}")

if __name__ == "__main__":
    main()
