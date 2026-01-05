# Application Streamlit MNIST - Version Basique

Application simplifiée pour prédire des chiffres manuscrits avec votre propre modèle CNN.

---

## 📋 Installation

1. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```

2. **Placer votre modèle** :
   - Copiez votre modèle `.keras` dans le dossier `models/`
   - Renommez-le en `mnist_model.keras`

---

## 🚀 Lancement

```bash
streamlit run app.py
```

L'application s'ouvrira dans votre navigateur (généralement http://localhost:8501)

---

## 📸 Utilisation

1. **Mode Upload** : Téléchargez une image de chiffre (PNG, JPG, JPEG)
2. **Mode Caméra** : Prenez une photo en direct

L'application affiche :
- Le chiffre prédit
- Le niveau de confiance
- Le top 3 des prédictions

---

## 🔧 Preprocessing appliqué

L'application applique automatiquement :
1. Conversion en niveaux de gris
2. Redimensionnement vers 28×28
3. Normalisation [0, 1] (division par 255)
4. Reshape vers (1, 28, 28, 1)

**Important** : Ce preprocessing suppose que votre modèle attend des images normalisées [0, 1].

---

## 📝 Format du modèle attendu

- **Format** : `.keras` (Keras 3.x)
- **Input shape** : `(batch_size, 28, 28, 1)`
- **Input type** : `float32`
- **Input range** : `[0, 1]`
- **Output shape** : `(batch_size, 10)`
- **Output** : Probabilités pour les classes 0-9 (softmax)

---

## 🛠️ Personnalisation

### Changer le nom du modèle

Modifiez la ligne 25 dans `app.py` :
```python
model_path = os.path.join(os.path.dirname(__file__), 'models', 'VOTRE_NOM.keras')
```

### Ajuster le preprocessing

Si votre modèle attend un format différent, modifiez la fonction `preprocess_image()` dans `app.py` (lignes 31-55).

---

## 📁 Structure

```
streamlit_basic/
├── app.py                  # Application principale
├── models/
│   └── mnist_model.keras  # Votre modèle (à placer ici)
├── requirements.txt        # Dépendances
└── README.md              # Ce fichier
```

---

## 💡 Conseils pour de meilleures prédictions

- Écrivez le chiffre en **noir** sur fond **blanc**
- Assurez-vous que le chiffre est **bien visible** et **net**
- **Centrez** le chiffre dans l'image
- Évitez les ombres et les reflets

---

## 🐛 Dépannage

### Le modèle n'est pas trouvé
- Vérifiez que le fichier est bien dans `models/`
- Vérifiez que le nom est exactement `mnist_model.keras`

### Erreur de shape
- Votre modèle doit accepter des inputs de shape `(batch_size, 28, 28, 1)`
- Vérifiez avec `model.summary()`

### Prédictions incorrectes
- Vérifiez que le preprocessing correspond à celui utilisé pendant l'entraînement
- Si votre modèle attend [0, 255] au lieu de [0, 1], supprimez le `/255.0` ligne 48

---

Bon développement ! 🚀
