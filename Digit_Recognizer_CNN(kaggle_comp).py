from google.colab import files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

uploaded = files.upload()

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X = train.drop('label', axis=1).values
y = train['label'].values

X = X / 255.0
test = test.values / 255.0

X = X.reshape(-1, 28, 28, 1)
test = test.reshape(-1, 28, 28, 1)

y = to_categorical(y)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax')) 

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val))

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')
plt.show()

preds = model.predict(test)
pred_labels = np.argmax(preds, axis=1)

submission = pd.DataFrame({
    'ImageId' : np.arange(1, len(pred_labels)+1),
    'Label' : pred_labels

})
submission.to_csv('1st_CNN.csv', index=False)
print("Submission ready!")

from google.colab import files
files.download('1st_CNN.csv')
