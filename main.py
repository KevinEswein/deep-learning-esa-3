import pickle

import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Laden des trainierten Modells
model = tf.keras.models.load_model('trained_model.h5')

# Modell erneut kompilieren
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Laden des Tokenizers
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

seq_length = 5  # Beispielwert, sollte mit Ihrem Modell übereinstimmen


def predict_next_word(model, tokenizer, text, seq_length):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=seq_length, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted)
    predicted_word = tokenizer.index_word[predicted_word_index]
    return predicted_word


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    next_word = predict_next_word(model, tokenizer, text, seq_length)
    return jsonify(predictions=[next_word])


if __name__ == '__main__':
    app.run(debug=True)
