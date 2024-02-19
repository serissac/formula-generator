import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Define some constants
OPERATORS = ['+', '-', '*', '/']
MAX_LENGTH = 10
NUM_SAMPLES = 10000

# Generate the dataset
def generate_dataset(num_samples):
    X = []
    y = []
    for _ in range(num_samples):
        a = np.random.randint(1, 100)
        b = np.random.randint(1, 100)
        operator = np.random.choice(OPERATORS)
        if operator == '+':
            result = a + b
        elif operator == '-':
            result = a - b
        elif operator == '*':
            result = a * b
        else:
            if b != 0:
                result = a / b
            else:
                result = a
        X.append(f"{a} {operator} {b}")
        y.append(str(result))
    return X, y

# Create the dataset
X, y = generate_dataset(NUM_SAMPLES)

# Tokenize the input sequences
tokenizer = Tokenizer(char_level=False)
tokenizer.fit_on_texts(X + y)
total_words = len(tokenizer.word_index) + 1

# Convert text sequences to sequences of integers
X_sequences = tokenizer.texts_to_sequences(X)
y_sequences = tokenizer.texts_to_sequences(y)

# Pad sequences to the same length
X_padded = pad_sequences(X_sequences, maxlen=MAX_LENGTH, padding='post')
y_padded = pad_sequences(y_sequences, maxlen=1, padding='post')

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words, 100, input_length=MAX_LENGTH),
    tf.keras.layers.LSTM(150),
    tf.keras.layers.Dense(total_words, activation='softmax')
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_padded, y_padded, epochs=10, batch_size=64)

# Function to generate a mathematical expression
def generate_expression(model, tokenizer, max_length):
    seed_text = np.random.choice(OPERATORS)  # Start with an operator
    result = ''
    for _ in range(max_length):
        encoded = tokenizer.texts_to_sequences([seed_text])[0]
        encoded = pad_sequences([encoded], maxlen=max_length, padding='post')
        y_pred = model.predict_classes(encoded, verbose=0)
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == y_pred:
                out_word = word
                break
        if out_word.isdigit() or out_word in OPERATORS:
            seed_text += ' ' + out_word
            result += out_word
        else:
            break
    return result

# Generate a mathematical expression
generated_expression = generate_expression(model, tokenizer, MAX_LENGTH)
print("Generated expression:", generated_expression)
