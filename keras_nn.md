### **Introduction to Keras**
Keras is an **open-source neural network library** built primarily for **Python**. It provides a high-level interface for deep learning, allowing users to **quickly build and train models** without requiring extensive knowledge of underlying computations.

### **Key Features:**
- **Developed by François Chollet**: Originally written to simplify neural network implementations.
- **Modular & Extensible**: Users can flexibly define layers, models, and optimization techniques.
- **Frontend & Backend Separation**:
  - **Frontend**: Keras provides a simple, unified interface.
  - **Backend**: Calls deep learning frameworks like **TensorFlow, Theano, and CNTK** for computation.

### **Integration with TensorFlow**
- Since **2017**, many Keras components have been integrated into **TensorFlow**.
- In **2019**, Keras became the **official high-level API** for **TensorFlow 2**.
- **tf.keras** is now the standard module for implementing deep learning in TensorFlow 2.

### **Why Use Keras?**
- **Easy to Use**: Simplifies neural network implementation.
- **Multi-Backend Support**: Works across different deep learning frameworks.
- **Seamless Integration** with TensorFlow 2 for **scalability and deployment**.

---

### **Common Functional Modules in Keras**
Keras provides a **high-level API** for building and training neural networks, offering a variety of **functional modules** to construct deep learning architectures. Based on the surrounding document content, which discusses **Keras Advanced API**, **model configuration**, and **training strategies**, here are the key points:

### **1. Common Network Layer Classes**
- **Tensor Mode (`tf.nn`)**: Provides low-level functions for neural network layers.
- **Layer-Based Approach (`tf.keras.layers`)**: Offers a variety of predefined layers, including:
  - **Dense Layer** (Fully Connected): Every neuron connects to all neurons in the previous layer.
    ```python
    from keras.layers import Dense
    dense_layer = Dense(units=64, activation='relu')
    ```
  - **Convolutional Layer** (`Conv2D`, `Conv3D`): Used for image and video processing.
    ```python
    from keras.layers import Conv2D
    conv_layer = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')
    ```

### **2. Network Container (`Sequential`)**
- **Encapsulates multiple layers** into a single model, simplifying forward propagation.
- **Example of a two-layer fully connected network**:
  ```python
  from tensorflow.keras import layers, Sequential
  network = Sequential([
      layers.Dense(3, activation=None),
      layers.ReLU(),
      layers.Dense(2, activation=None),
      layers.ReLU()
  ])
  x = tf.random.normal([4,3])
  out = network(x)
  ```
- **Dynamically adding layers using `add()`**:
  ```python
  layers_num = 2
  network = Sequential([])
  for _ in range(layers_num):
      network.add(layers.Dense(3))
      network.add(layers.ReLU())
  network.build(input_shape=(4, 4))
  network.summary()
  ```

---

This section covers **Model Configuration, Training, and Testing in Keras**, providing a structured approach to building and evaluating deep learning models. Based on the surrounding document content, which discusses **Keras Advanced API**, **model optimization**, and **training strategies**, here are the key points:

### **1. Model Creation and Configuration**
- **Define the architecture** using `Sequential()` or Functional API.
- **Example:**
  ```python
  from keras.models import Sequential
  from keras.layers import Dense

  model = Sequential()
  model.add(Dense(units=64, activation='relu', input_dim=feature_dim))
  model.add(Dense(units=num_classes, activation='softmax'))
  ```

### **2. Compiling the Model**
- **Specify optimizer, loss function, and evaluation metrics**:
  ```python
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  ```

### **3. Data Preparation**
- **Split dataset into training and testing sets**:
  ```python
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
  ```

### **4. Model Training**
- **Train the model using `fit()`**:
  ```python
  history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
  ```

### **5. Model Evaluation**
- **Assess performance on test data**:
  ```python
  test_loss, test_accuracy = model.evaluate(X_test, y_test)
  print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
  ```

### **6. Making Predictions**
- **Use trained model for inference**:
  ```python
  predictions = model.predict(X_new_data)
  ```

### **7. Saving and Loading Models (Optional)**
- **Save model for future use**:
  ```python
  model.save("my_model.h5")
  ```
- **Load saved model**:
  ```python
  loaded_model = keras.models.load_model("my_model.h5")
  ```

### **8. Visualization (Optional)**
- **Plot training history using Matplotlib**:
  ```python
  import matplotlib.pyplot as plt
  plt.plot(history.history['loss'], label='Training Loss')
  plt.plot(history.history['val_loss'], label='Validation Loss')
  plt.legend()
  plt.show()
  ```

---

### **Model Saving and Loading** in Keras 
### **1. Tensor Method (`tf.saved_model`)**
- **Purpose:** Saves the entire model, including architecture, weights, and metadata, making it compatible with **TensorFlow Serving**.
- **Saving a Model:**
  ```python
  model.save('my_model')
  ```
  - Creates a directory (`my_model`) containing all model artifacts.
- **Loading a Model:**
  ```python
  loaded_model = tf.saved_model.load('my_model')
  predictions = loaded_model(X_new_data)
  ```
  - Loads the model and allows inference on new data.

### **2. Network Method (`model.save(path)`)**
- **Purpose:** Saves only the model parameters (`.h5` format), allowing recovery without needing the original network source files.
- **Saving a Model:**
  ```python
  network.save('model.h5')
  del network  # Delete the network object
  ```
- **Loading a Model:**
  ```python
  network = keras.models.load_model('model.h5')
  ```
  - Restores the model **without manually recreating the architecture**.

### **3. SavedModel Method (`tf.saved_model.save`)**
- **Purpose:** Platform-independent model saving, useful for **mobile and web deployment**.
- **Saving a Model:**
  ```python
  tf.saved_model.save(network, 'model-savedmodel')
  del network  # Delete network object
  ```
- **Loading and Evaluating:**
  ```python
  network = tf.saved_model.load('model-savedmodel')
  acc_meter = metrics.CategoricalAccuracy()
  for x, y in ds_val:
      pred = network(x)
      acc_meter.update_state(y_true=y, y_pred=pred)
  print("Test Accuracy:%f" % acc_meter.result())
  ```
  - Loads the model and computes accuracy on a test dataset.

---

### **Custom Network Layers** n Keras
    creating specialized neural network architectures by subclassing `keras.Model`.
### **1. Custom Layer Creation**
In Keras, you can define custom layers by subclassing `keras.layers.Layer`. Each custom layer requires:
- An **initializer (`__init__`)** to define parameters like number of units and activation functions.
- A **build method (`build`)** to create layer-specific weights (like `self.kernel` and `self.bias`).
- A **call method (`call`)** for the forward pass logic.

**Example:**
```python
class CustomLayer(Layer):
    def __init__(self, num_units, activation='relu'):
        super(CustomLayer, self).__init__()
        self.num_units = num_units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel", shape=[input_shape[-1], self.num_units])
        self.bias = self.add_weight("bias", shape=[self.num_units])

    def call(self, inputs):
        return self.activation(tf.matmul(inputs, self.kernel) + self.bias)
```

### **2. Creating a Custom Model**
To build a full network, subclass `keras.Model` and stack multiple custom layers:
```python
class CustomModel(tf.keras.Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.layer1 = CustomLayer(num_units=64, activation='relu')
        self.layer2 = CustomLayer(num_units=10, activation='softmax')

    def call(self, inputs):
        x = self.layer1(inputs)
        return self.layer2(x)
```
This allows precise control over **forward propagation** logic.

### **3. Compiling & Training**
Once the model is set up, it needs to be compiled and trained:
```python
model = CustomModel()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
```
- Uses **Adam optimizer** for efficient gradient updates.
- Loss function set to **categorical cross-entropy** (for classification tasks).

### **4. Using the Model**
After training, evaluate and make predictions:
```python
loss, accuracy = model.evaluate(X_test, y_test)
predictions = model.predict(X_new_data)
```
This treats the custom model like any other Keras model.

---

### **Model Zoo**
    provides easy access to commonly used **pre-trained deep learning models** like **ResNet** and **VGG** via `keras.applications`.

### **1. Loading Pre-Trained Models**
- Instead of manually defining architectures, you can directly load models such as **ResNet50** using:
  ```python
  resnet = keras.applications.ResNet50(weights='imagenet', include_top=False)
  resnet.summary()
  ```
- The `weights='imagenet'` parameter ensures the model is pre-trained on the **ImageNet dataset**.
- `include_top=False` removes the final classification layer, allowing customization.

### **2. Modifying for Custom Tasks**
For specialized tasks (e.g., classifying 100 objects instead of ImageNet's 1000):
- A **pooling layer** reduces feature dimensions, making processing more efficient:
  ```python
  global_average_layer = layers.GlobalAveragePooling2D()
  out = global_average_layer(x)
  ```
- A **fully connected (Dense) layer** sets the number of output nodes:
  ```python
  fc = layers.Dense(100)
  out = fc(x)
  ```

### **3. Building a Custom Model**
To integrate these layers into a new model:
```python
mynet = Sequential([resnet, global_average_layer, fc])
mynet.summary()
```
This approach **reuses** ResNet50 while adjusting the architecture for new tasks.

### **4. Freezing Parameters for Efficient Training**
By setting:
```python
resnet.trainable = False
```
- The **pre-trained weights** are frozen, preventing unnecessary updates.
- Only the **new layers** get trained, **reducing computational cost** and training time.

---

### **Metrics in Keras**
    tracking model performance during training and evaluation. 
### **1. Creating a Metrics Container**
- Keras provides built-in metric classes in `keras.metrics`, including:
  - **Mean (`metrics.Mean()`)** – Tracks the average loss.
  - **Accuracy (`metrics.Accuracy()`)** – Measures how often predictions match labels.
  - **Cosine Similarity (`metrics.CosineSimilarity()`)** – Computes similarity between predictions and ground truth.

### **2. Writing Data (`update_state()`)**
- Metrics accumulate new data through the `update_state()` method:
  ```python
  loss_meter = metrics.Mean()
  loss_meter.update_state(float(loss))
  ```

### **3. Reading Statistical Data (`result()`)**
- Retrieve the computed metric:
  ```python
  print(step, 'loss:', loss_meter.result())
  ```

### **4. Clearing the Container (`reset_states()`)**
- Since metrics record historical data, reset them before a new round:
  ```python
  if step % 100 == 0:
      print(step, 'loss:', loss_meter.result())
      loss_meter.reset_states()
  ```

### **5. Hands-On Accuracy Metric**
- Create an accuracy tracker:
  ```python
  acc_meter = metrics.Accuracy()
  ```
- Record predictions:
  ```python
  pred = tf.argmax(network(x), axis=1)
  pred = tf.cast(pred, dtype=tf.int32)
  acc_meter.update_state(y, pred)
  ```
- Retrieve accuracy and reset:
  ```python
  print(step, 'Evaluate Acc:', acc_meter.result().numpy())
  acc_meter.reset_states()
  ```

---

### **Visualization in Keras**
    monitoring training progress and model performance using **Python scripts** and **browser-based tools** like TensorBoard.

### **1. Model-Side Visualization**
- **Tracking Metrics (`tf.summary.create_file_writer`)**:
  ```python
  summary_writer = tf.summary.create_file_writer(log_dir)
  ```
- **Logging Loss (`tf.summary.scalar`)**:
  ```python
  with summary_writer.as_default():
      tf.summary.scalar('train-loss', float(loss), step=step)
  ```
- **Logging Images (`tf.summary.image`)**:
  ```python
  with summary_writer.as_default():
      tf.summary.image("val-onebyone-images:", val_images, max_outputs=9, step=step)
  ```

### **2. Plot Training History**
- **Using Matplotlib to visualize loss and accuracy**:
  ```python
  import matplotlib.pyplot as plt
  plt.plot(history.history['loss'], label='Training Loss')
  plt.plot(history.history['val_loss'], label='Validation Loss')
  plt.legend()
  plt.show()
  ```

### **3. Model Summary**
- **Print model architecture**:
  ```python
  model.summary()
  ```

### **4. Browser-Side Visualization (TensorBoard)**
- **Start TensorBoard**:
  ```bash
  tensorboard --logdir=./logs
  ```
- **Access via browser (`http://localhost:6006`)** to view:
  - **SCALARS** (loss, accuracy)
  - **IMAGES** (sample visualizations)
  - **HISTOGRAMS** (tensor distributions)

### **5. Logging Metrics to TensorBoard**
- **Using Keras callbacks**:
  ```python
  from tensorflow.keras.callbacks import TensorBoard
  tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)
  model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[tensorboard_callback])
  ```
