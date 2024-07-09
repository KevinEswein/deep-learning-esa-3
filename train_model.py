import pickle

import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


# Laden der Textdaten
def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text


# Tokenisierung und Sequenzierung der Textdaten
def prepare_sequences(text, seq_length):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    sequences = []

    words = text.split()
    for i in range(seq_length, len(words)):
        seq = words[i - seq_length:i + 1]
        encoded_seq = tokenizer.texts_to_sequences([seq])[0]
        sequences.append(encoded_seq)

    sequences = pad_sequences(sequences, maxlen=seq_length + 1, padding='pre')
    return tokenizer, sequences


# Modellarchitektur
def create_model(vocab_size, seq_length):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=50, input_length=seq_length))
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(vocab_size, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
    return model


# Training
def train_model(model, sequences, epochs=50, batch_size=32):
    X = sequences[:, :-1]
    y = sequences[:, -1]

    # Dummy Pass to build the model
    model.build(input_shape=(None, X.shape[1]))

    y = tf.keras.utils.to_categorical(y, num_classes=model.output_shape[-1])

    model.fit(X, y, epochs=epochs, batch_size=batch_size)
    return model


# Laden und Vorbereiten der Daten
text = load_text('data.txt')
seq_length = 5  # Beispielwert
tokenizer, sequences = prepare_sequences(text, seq_length)

# Modell erstellen und trainieren
model = create_model(vocab_size=len(tokenizer.word_index) + 1, seq_length=seq_length)
model = train_model(model, sequences, epochs=50, batch_size=32)

# Modell und Tokenizer speichern
model.save('trained_model.h5')
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
