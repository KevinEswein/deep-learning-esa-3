import json
import pickle

with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Konvertieren des Tokenizers in ein JSON-kompatibles Format
tokenizer_json = tokenizer.to_json()

# Speichern des Tokenizers
with open('tokenizer.json', 'w') as f:
    json.dump(tokenizer_json, f)
