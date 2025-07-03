# Tweet Sentiment Prediction: A Comparative Approach with RapidMiner and Python

## Project Description
This project focuses on predicting sentiments in text, specifically in tweets categorized by emotions. The goal is to automatically analyze and classify emotions expressed in tweets using text mining techniques and machine learning algorithms. The performance of different approaches and models implemented in both RapidMiner and Python are compared.

## Objectives
* Apply advanced text analysis processes to a tweet dataset classified into six main emotions (sadness, joy, love, anger, fear, surprise).
* Implement data preprocessing and classification models in RapidMiner, including Random Forest and Deep Learning.
* Utilize advanced language representations like Word Embeddings (ConceptNet Numberbatch, GloVe, BERT) in Python for improved predictive capability.
* Compare model performance in terms of precision, recall, and accuracy.

## Dataset
The dataset used is a CSV file containing tweets in English, with relevant information for sentiment prediction.
Each row represents a specific tweet, with the following fields:
* **Text**: Full text of the tweet.
* **Label**: Number representing the emotion associated with the tweet, categorized as follows:
    * 0: Sadness
    * 1: Joy
    * 2: Love
    * 3: Anger
    * 4: Fear
    * 5: Surprise
The dataset was balanced to 6000 instances per class, totaling approximately 36000 instances.

## Methodology

### RapidMiner
The first stage of the study was conducted in RapidMiner, focusing on data preprocessing and the implementation of classic models.

#### Data Preparation in RapidMiner
* **Attribute Selection**: `Text` and `Label` were selected. A new `Label_text` variable was created for textual descriptions of emotions.
* **Text Conversion**: Operations such as tokenization, stopword removal (English only), lemmatization or stemming (Snowball algorithm), and vector generation (TF-IDF) were performed.
* **Token Filtering**: A token length filter (minimum 3, maximum 25 characters) and a pruning method were applied to remove irrelevant terms.
* **Class Balancing**: The `Sample` operator was used to balance categories to 6000 instances per class.

#### Models in RapidMiner
4 Deep Learning models and 1 Random Forest model were created.
* **Model 1 and 5 (Base)**: Basic preprocessing.
* **Model 2**: Includes sentiment analysis (VADER model).
* **Model 3**: Adds n-gram creation.
* **Model 4**: Combines n-grams with sentiment analysis.

### Python Implementation
In the second stage, advanced language representations and models were explored in Python.

#### Data Preparation in Python
* **Reading and Filtering**: The same CSV file was used, and a limit of 5000 records per class was applied for dataset balancing.
* **Data Splitting**: Data was divided into training and test subsets.

#### Word Embeddings
The following Word Embedding techniques were used to capture semantic and contextual relationships:
* **ConceptNet Numberbatch**: Static, based on semantic relations.
* **GloVe (Global Vectors for Word Representation)**: Static, based on textual co-occurrences.
* **BERT (Bidirectional Encoder Representations from Transformers)**: Dynamic and contextual.

#### Models in Python
Several machine learning models were evaluated, starting from the best results obtained in RapidMiner (Deep Learning and Random Forest). Other algorithms explored include:
* K-Nearest Neighbors (KNN)
* Gradient Boosting
* Random Forest
* Artificial Neural Networks (ANN)

**Artificial Neural Network (ANN) Architecture**
The model follows a dense architecture with several connected layers, using activation functions (ReLU) and batch normalization (BatchNorm1d) to improve training. Dropout was implemented to prevent overfitting.

## Results
In RapidMiner, models that used "extract sentiment" (Model 2 and Model 4) showed the best results. For Python implementations, the models trained with the neural network generally outperformed those trained with Random Forest, with the ConceptNet-RNA model yielding the best results overall.

## Reminder
The data and word embeddings utilized in this project are not available within this repository. This is due to their substantial file sizes, which exceed the limits permissible by GitHub for direct inclusion.

## Contributors
* Amina Errami Maslaoui
* Aitana Antonia Ortiz Guiño
* Martín Portugal Gónzalez
* Alba Vidales Casado

## License
*No license is specified in the original document. It is recommended to add an appropriate license for GitHub projects.*
