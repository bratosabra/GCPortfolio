# GCPortfolio
GCPortfolio is a collection of various projects created using Google Colaboratory. This repository serves as a showcase of my work, experiments, and learning journey with data science, machine learning, deep learning, and other related fields.

---

## **Sentiment Analysis using IMDB Dataset**  
This project performs sentiment analysis on movie reviews using a Long Short-Term Memory (LSTM) neural network. The goal is to classify movie reviews as **positive** or **negative** based on their content.

---

### **Overview**  
- **Dataset**: IMDB dataset (pre-built in Keras).  
- **Model**: LSTM (Long Short-Term Memory).  
- **Tools**: TensorFlow, Keras, Python.  
- **Input**: Preprocessed movie reviews.  
- **Output**: Sentiment prediction (Positive/Negative).  

---

### **Features**  
1. **LSTM Model**: A sequential LSTM network for natural language processing tasks.  
2. **Ready-to-use Dataset**: IMDB dataset is loaded directly via `Keras.datasets`.  
3. **Visualization**: Training and validation accuracy plotted over epochs.  
4. **Custom Input**: Users can input their own text for real-time sentiment analysis.

---

### **Dependencies**  
Make sure you have the following libraries installed:

```bash
pip install tensorflow numpy matplotlib
```

---

### **How to Run**  

1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/imdb-sentiment-analysis.git
   cd imdb-sentiment-analysis
   ```

2. Open the file in Google Colab or any Python IDE.  

3. Run the code step-by-step.  

4. To predict sentiment on custom text, input a sentence when prompted.  

---

### **Code Explanation**  

1. **Load the Dataset**  
   The IMDB dataset contains 25,000 labeled movie reviews for training and testing.  
   ```python
   from tensorflow.keras.datasets import imdb
   (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
   ```

2. **Preprocessing**  
   The text data is padded to ensure uniform input size.  
   ```python
   from tensorflow.keras.preprocessing import sequence
   x_train = sequence.pad_sequences(x_train, maxlen=200)
   x_test = sequence.pad_sequences(x_test, maxlen=200)
   ```

3. **LSTM Model**  
   The LSTM network processes the sequence data and outputs a sentiment prediction.  
   ```python
   model = tf.keras.Sequential([
       tf.keras.layers.Embedding(10000, 128, input_length=200),
       tf.keras.layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
       tf.keras.layers.Dense(1, activation='sigmoid')
   ])
   ```

4. **Training and Evaluation**  
   The model is trained using binary cross-entropy loss and evaluated on test data.  
   ```python
   model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
   model.fit(x_train, y_train, epochs=3, batch_size=32, validation_data=(x_test, y_test))
   ```

5. **Predict Custom Input**  
   Users can enter any text, and the model will predict whether the sentiment is **positive** or **negative**.  

---

### **Example Input/Output**  

**Input**:  
`"This movie is fantastic, I loved every part of it!"`  

**Output**:  
`Sentiment: Positive`

---

### **Results**  
- Achieved high accuracy on the IMDB dataset.  
- Successfully predicts sentiment on custom inputs.  

---

### **Screenshots**  
Include screenshots for visualization:  
![Без названия](https://github.com/user-attachments/assets/f88168fc-bd72-45f6-86d9-fac115318703)

  

