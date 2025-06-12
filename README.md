# Deep Sentiment Analysis with Word Embeddings 

This project uses Keras and TensorFlow to build a deep learning model for sentiment classification. It shows how to apply an Embedding layer and RNN (Recurrent Neural Network) to analyze natural language reviews.

##  Objectives

- Understand how Keras Embedding layers work
- Prepare a labeled sentiment dataset for deep learning
- Train a Sequential model with embeddings and dense layers
- Build a more advanced model using RNN
- Evaluate and visualize model performance

##  Tech Stack

- Python
- TensorFlow / Keras
- NumPy, Pandas
- NLTK
- Matplotlib

##  Dataset

- Initial small dataset defined manually for demonstration
- Extended IMDB or review dataset can be integrated using `pandas`

##  Key Steps

1. **Text Preprocessing**:
   - Tokenization
   - Stopword removal
   - Padding sequences

2. **Model Building**:
   - Keras `Embedding` layer
   - `Flatten` or `SimpleRNN` for feature extraction
   - Output layer with sigmoid activation

3. **Training & Evaluation**:
   - Binary cross-entropy loss
   - Accuracy tracking
   - Visualization of loss and accuracy curves

