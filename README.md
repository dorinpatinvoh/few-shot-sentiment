# Analyse de Sentiments Few-Shot avec MAML et CamemBERT

Ce projet implémente un système d'analyse de sentiments en français utilisant l'apprentissage few-shot avec Model-Agnostic Meta-Learning (MAML) et CamemBERT.

## Description

Le système utilise :
- CamemBERT comme modèle de base pour le traitement du texte en français
- MAML (Model-Agnostic Meta-Learning) pour l'apprentissage few-shot
- Une approche d'apprentissage auto-supervisé pour améliorer les performances

## Installation

```bash
# Cloner le repository
git clone https://github.com/dorinpatinvoh/few-shot-sentiment.git
cd few-shot-sentiment

# Installer les dépendances
pip install torch transformers
```

## Utilisation

```python
# Exemple d'utilisation
model = MAMLFewShotLearner(model_name="camembert-base")

# Test sur des exemples
test_texts = ["Ce produit est fantastique!", "Service client déplorable"]
test_labels = [1, 0]

test_batch = prepare_few_shot_batch(test_texts, test_labels, model.tokenizer)
predictions = model.predict(test_batch)
```

## Structure du Projet

- `main.py` : Script principal contenant l'implémentation
- Composants principaux :
  - `FewShotLearner` : Classe de base pour l'apprentissage few-shot
  - `MAMLFewShotLearner` : Implémentation MAML
  - `DatasetManager` : Gestion des données d'entraînement et de test

## Auteur

- Dorin PATINVOH
