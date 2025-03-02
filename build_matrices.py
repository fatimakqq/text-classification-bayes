
import os
import re
import csv
import nltk
from nltk.corpus import stopwords
from collections import Counter

# Download NLTK resources if not already downloaded
nltk.download('stopwords')

def preprocess_text(text):
    """
    Preprocess text by converting to lowercase and removing punctuation
    """
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    return text

def extract_words(file_path):
    """
    Extract words from a file after preprocessing
    """
    with open(file_path, 'r', encoding='latin-1') as file:
        text = file.read()
    
    # Preprocess text
    text = preprocess_text(text)
    
    # Tokenize
    words = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    return words

def build_vocabulary(train_spam_dir, train_ham_dir):
    """
    Build vocabulary from training data
    """
    vocabulary = set()
    
    # Process spam files
    for filename in os.listdir(train_spam_dir):
        file_path = os.path.join(train_spam_dir, filename)
        words = extract_words(file_path)
        vocabulary.update(words)
    
    # Process ham files
    for filename in os.listdir(train_ham_dir):
        file_path = os.path.join(train_ham_dir, filename)
        words = extract_words(file_path)
        vocabulary.update(words)
    
    return sorted(list(vocabulary))

def create_bow_vector(file_path, vocabulary):
    """
    Create Bag of Words vector for a file
    """
    words = extract_words(file_path)
    word_counts = Counter(words)
    
    # Create vector
    vector = [word_counts.get(word, 0) for word in vocabulary]
    
    return vector

def create_bernoulli_vector(file_path, vocabulary):
    """
    Create Bernoulli vector for a file
    """
    words = extract_words(file_path)
    word_set = set(words)
    
    # Create vector
    vector = [1 if word in word_set else 0 for word in vocabulary]
    
    return vector

def create_feature_matrix(data_dir, is_spam, vocabulary, representation):
    """
    Create feature matrix for files in a directory
    """
    matrix = []
    
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        
        if representation == 'bow':
            vector = create_bow_vector(file_path, vocabulary)
        else:  # bernoulli
            vector = create_bernoulli_vector(file_path, vocabulary)
        
        # Add label (1 for spam, 0 for ham)
        vector.append(1 if is_spam else 0)
        matrix.append(vector)
    
    return matrix

def save_matrix_to_csv(matrix, vocabulary, output_file):
    """
    Save feature matrix to CSV file
    """
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header (vocabulary words + label)
        header = vocabulary + ['label']
        writer.writerow(header)
        
        # Write data
        writer.writerows(matrix)

def process_dataset(dataset_name, representation):
    """
    Process a dataset and create feature matrices
    """
    base_dir = f'./{dataset_name}'
    train_spam_dir = os.path.join(base_dir, 'train', 'spam')
    train_ham_dir = os.path.join(base_dir, 'train', 'ham')
    test_spam_dir = os.path.join(base_dir, 'test', 'spam')
    test_ham_dir = os.path.join(base_dir, 'test', 'ham')
    
    print(f"Processing {dataset_name} with {representation} representation...")
    
    # Build vocabulary from training data
    vocabulary = build_vocabulary(train_spam_dir, train_ham_dir)
    print(f"Vocabulary size: {len(vocabulary)}")
    
    # Create training matrix
    train_spam_matrix = create_feature_matrix(train_spam_dir, True, vocabulary, representation)
    train_ham_matrix = create_feature_matrix(train_ham_dir, False, vocabulary, representation)
    train_matrix = train_spam_matrix + train_ham_matrix
    
    # Create test matrix
    test_spam_matrix = create_feature_matrix(test_spam_dir, True, vocabulary, representation)
    test_ham_matrix = create_feature_matrix(test_ham_dir, False, vocabulary, representation)
    test_matrix = test_spam_matrix + test_ham_matrix
    
    # Save matrices to CSV
    train_output = f"{dataset_name}_{representation}_train.csv"
    test_output = f"{dataset_name}_{representation}_test.csv"
    
    save_matrix_to_csv(train_matrix, vocabulary, train_output)
    save_matrix_to_csv(test_matrix, vocabulary, test_output)
    
    print(f"Saved {train_output} and {test_output}")

# Process all datasets with both representations
datasets = ['enron1', 'enron2', 'enron4']
representations = ['bow', 'bernoulli']

for dataset in datasets:
    for representation in representations:
        process_dataset(dataset, representation)