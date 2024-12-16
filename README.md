# GCPortfolio
GCPortfolio is a collection of various projects created using Google Colaboratory. This repository serves as a showcase of my work, experiments, and learning journey with data science, machine learning, deep learning, and other related fields.

---
1. [Sentiment Analysis using IMDB Dataset](#Sentiment-Analysis-using-IMDB-Dataset)
2. [Handwritten Digit Recognition using MNIST Dataset](#Handwritten-Digit-Recognition-using-MNIST-Dataset)
3. [Titanic-Survival-EDA](Titanic-Survival-EDA)
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
![Без названия (1)](https://github.com/user-attachments/assets/0f08a7be-e301-4734-b3bd-cb58b5933754)


---

## **Handwritten Digit Recognition using MNIST Dataset**

### **Overview**
This project focuses on building a simple neural network to recognize handwritten digits (0–9) using the MNIST dataset. The model predicts the digit shown in an input image with high accuracy.

---

### **Features**
1. **Dataset**: MNIST, consisting of 70,000 grayscale images of handwritten digits.
2. **Model**: A simple feedforward neural network with two hidden layers.
3. **Accuracy**: Achieves high accuracy on test data with minimal computation.
4. **Visualization**: Includes model training graphs and prediction samples.

---

### **Technologies Used**
- **Python**: Programming language.
- **TensorFlow/Keras**: Neural network framework.
- **NumPy**: Data processing.
- **Matplotlib**: Data visualization.

---

### **Dependencies**
Install the required libraries with the following command:
```bash
pip install tensorflow numpy matplotlib
```

---

### **How to Run the Project**

1. Clone the repository:

2. Open the notebook file in **Google Colab** or a local Jupyter environment.

3. Run the code step-by-step:
   - Load the dataset.
   - Train the neural network.
   - Evaluate its performance.

4. Test the model by running predictions on random or custom inputs.

---

### **Code Highlights**
- **Preprocessing**: Normalize input images and apply one-hot encoding for labels.
- **Model**: A simple sequential model with `Dense` layers.
- **Training**: The model is trained with the `adam` optimizer and categorical cross-entropy loss.
- **Visualization**: Graphs for training accuracy and loss, along with example predictions.

---

### **Example Input and Output**

**Input**: A grayscale image of a handwritten digit.  
**Output**: Predicted digit with its confidence score.

**Sample Visualization**:
```plaintext
Input Image: (28x28 pixels)
Model Prediction: 7 (Confidence: 98.6%)
```

---

### **Project Workflow**

1. **Dataset Preparation**:
   - Load MNIST dataset from Keras.
   - Normalize pixel values (0–255 scaled to 0–1).
   - Convert labels to categorical format.

2. **Model Creation**:
   - Use a `Flatten` layer to preprocess input.
   - Add two `Dense` layers with ReLU activation.
   - Add a final output layer with `softmax` activation.

3. **Training**:
   - Train the model for 5 epochs.
   - Use 20% of training data for validation.

4. **Evaluation**:
   - Test the model on unseen data.
   - Display accuracy and example predictions.

---

### **Results**
- **Training Accuracy**: ~98% after 5 epochs.
- **Test Accuracy**: ~97%.

---

### **Screenshots**
Include plots and prediction examples like:   
![Без названия (2)](https://github.com/user-attachments/assets/37bb9172-e8ea-4731-a4b8-1582a781ca0e)
![Без названия (3)](https://github.com/user-attachments/assets/c20154ed-06a5-478b-934f-295791502600)

---

## Titanic-Survival-EDA
### **Overview**
This project explores the famous Titanic dataset to analyze survival rates based on various features such as age, gender, class, and more. The goal is to use Exploratory Data Analysis (EDA) techniques to generate insights and visualize key patterns in the data.


### Key Features
- **Data Cleaning:** Handling missing values and encoding categorical variables.
- **Data Visualization:** Using libraries like Matplotlib and Seaborn to create plots and heatmaps.
- **Statistical Insights:** Generating summaries and survival statistics by different groups.

### Dataset
The Titanic dataset is loaded from the `Seaborn` library. Alternatively, you can use the dataset from Kaggle: [Titanic Dataset](https://www.kaggle.com/c/titanic).

### Tools and Libraries
- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn

### Project Workflow
1. **Data Loading:** Importing the Titanic dataset.
2. **Data Cleaning:**
   - Filling missing values.
   - Dropping irrelevant columns.
   - Encoding categorical variables.
3. **Exploratory Data Analysis (EDA):**
   - Visualizing distributions of numerical and categorical features.
   - Analyzing survival rates based on features like age, gender, and class.
   - Generating heatmaps to identify correlations.
4. **Conclusion:** Summarizing key findings from the analysis.

### Visualizations
- **Age Distribution:** Histogram with Kernel Density Estimation (KDE).
- **Survival by Gender:** Bar plot.
- **Survival by Class:** Count plot.
- **Heatmap of Correlations:** Visualizing relationships between variables.

### Example Plots
![Без названия (1)](https://github.com/user-attachments/assets/9b907943-ddb1-4339-b4d0-c9e17bbd75d4)   
*Histogram showing the distribution of passengers' ages.*

![Без названия](https://github.com/user-attachments/assets/a26bae56-9164-49e9-8b41-3029e7d6741c)   
*Count plot of survival rates by passenger class.*

