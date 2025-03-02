
import os
import re
import csv
import nltk
from nltk.corpus import stopwords
from collections import Counter

nltk.download('stopwords')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def extract_words(file_path):
    #open file
    with open(file_path, 'r', encoding='latin-1') as file:
        text = file.read()
    
    text = clean_text(text) #clean text
    words = text.split() #tokenization
    
    #filter out stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = []
    for word in words:
        if word not in stop_words:
            filtered_words.append(word)

    return filtered_words

def build_vocab(train_spam_path, train_ham_path):
    vocab = set()
    
    #process spam files
    for filename in os.listdir(train_spam_path):
        file_path= os.path.join(train_spam_path, filename)
        words = extract_words(file_path)
        vocab.update(words)
    
    #process ham files
    for filename in os.listdir(train_ham_path):
        file_path = os.path.join(train_ham_path, filename)
        words =extract_words(file_path)
        vocab.update(words)

    res = sorted(list(vocab))
    return res

def create_bow_vector(file_path, vocab): #BAG OF WORDS VECTOR
    words = extract_words(file_path)
    word_counts = Counter(words)
    
    v = [word_counts.get(word, 0) for word in vocab]
    return v

def create_bernoulli_vector(file_path, vocab): #BERNOULLI VECTOR
    words = extract_words(file_path)
    word_set = set(words)
    
    v = [1 if word in word_set else 0 for word in vocab]
    return v

def create_feature_matrix(data_path, is_spam, vocab, representation): #build matrix
    matrix = []
    
    for filename in os.listdir(data_path):
        file_path = os.path.join(data_path, filename) #get file path
        
        if representation == 'bow':
            v = create_bow_vector(file_path, vocab)
        else:  #bernoulli
            v = create_bernoulli_vector(file_path, vocab)
        
        v.append(1 if is_spam else 0)
        matrix.append(v)
    
    return matrix

def save_matrix_to_csv(matrix, vocab, output_file): #BUILD CSV file
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        #column names
        header = vocab + ['label']
        writer.writerow(header)
        
        writer.writerows(matrix) #write all data rows

def process_dataset(dataset_name, representation):
    base_path = f'./{dataset_name}'

    train_spam_path = os.path.join(base_path, 'train', 'spam')
    test_spam_path = os.path.join(base_path, 'test', 'spam')

    train_ham_path = os.path.join(base_path, 'train', 'ham')
    test_ham_path = os.path.join(base_path, 'test', 'ham')
        
    vocab = build_vocab(train_spam_path, train_ham_path) #build vocab on training data
    #print(f"vocab: {len(vocab)}")
    
    # matrices from TRAINING data
    train_spam_matrix = create_feature_matrix(train_spam_path, True, vocab, representation)
    train_ham_matrix = create_feature_matrix(train_ham_path, False, vocab, representation)
    train_matrix = train_spam_matrix + train_ham_matrix
    
    # create matrices from TEST data
    test_spam_matrix = create_feature_matrix(test_spam_path, True, vocab, representation)
    test_ham_matrix = create_feature_matrix(test_ham_path, False, vocab, representation)
    test_matrix = test_spam_matrix + test_ham_matrix
    
    #write matrices to csv files
    # naming conventiion: dataset_representation_set.csv
    
    train_output = f"{dataset_name}_{representation}_train.csv"
    save_matrix_to_csv(train_matrix, vocab, train_output)

    test_output = f"{dataset_name}_{representation}_test.csv"
    save_matrix_to_csv(test_matrix, vocab, test_output)
    

#call main processing function
datasets = ['enron1', 'enron2', 'enron4']
representations = ['bow', 'bernoulli']

for dataset in datasets:
    for representation in representations:
        process_dataset(dataset, representation)