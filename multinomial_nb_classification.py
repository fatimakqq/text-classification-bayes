import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class MultinomialNaiveBayes:
    def __init__(self):
        self.class_priors = None
        self.feature_probs = None
    
    def fit(self, X, y): #train the classifier
        # x is the feature matrix 
        # y is the label: spam vs ham

        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        #calculate class priors
        self.class_priors = np.zeros(n_classes)
        for i, c in enumerate(self.classes):
            self.class_priors[i] = np.sum(y == c) / n_samples
        
        # Calculate word probabilities with Laplace smoothing
        self.feature_probs = np.zeros((n_classes, n_features))
        
        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            
            # Calculate the total count of words for this class
            total_count = np.sum(X_c) + n_features  # Add n_features for Laplace smoothing
            
            # Calculate probability for each word with Laplace smoothing
            for j in range(n_features):
                word_count = np.sum(X_c[:, j]) + 1  # Add 1 for Laplace smoothing
                self.feature_probs[i, j] = word_count / total_count
        
        # Convert to log probabilities
        self.log_class_priors = np.log(self.class_priors)
        self.log_feature_probs = np.log(self.feature_probs)
    
    def predict(self, X):
        """
        Predict class labels for samples in X
        
        Parameters:
        X: Feature matrix (BoW representation)
        
        Returns:
        y_pred: Predicted class labels
        """
        return np.array([self._predict_sample(x) for x in X])
    
    def _predict_sample(self, x):
        """
        Predict class label for a single sample
        
        Parameters:
        x: Feature vector (BoW representation)
        
        Returns:
        predicted_class: Predicted class label
        """
        posteriors = []
        
        # Calculate posterior probability for each class
        for i, c in enumerate(self.classes):
            # Start with log prior
            log_posterior = self.log_class_priors[i]
            
            # Add log likelihood for each feature
            # For BoW, we multiply the log probabilities by the word count
            log_posterior += np.sum(x * self.log_feature_probs[i])
            
            posteriors.append(log_posterior)
        
        # Return the class with the highest posterior probability
        return self.classes[np.argmax(posteriors)]

def load_data(csv_file):
    """
    Load data from CSV file
    
    Returns:
    X: Feature matrix
    y: Labels
    """
    data = pd.read_csv(csv_file)
    y = data['label'].values
    X = data.drop('label', axis=1).values
    return X, y

def evaluate_model(y_true, y_pred):
    """
    Evaluate model performance
    
    Returns:
    metrics: Dictionary of evaluation metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }
    return metrics

def run_multinomial_nb(dataset):
    """
    Run Multinomial Naive Bayes on BoW dataset
    
    Parameters:
    dataset: Dataset name (enron1, enron2, or enron4)
    
    Returns:
    metrics: Dictionary of evaluation metrics
    """
    train_file = f"{dataset}_bow_train.csv"
    test_file = f"{dataset}_bow_test.csv"
    
    # Load data
    X_train, y_train = load_data(train_file)
    X_test, y_test = load_data(test_file)
    
    # Train model
    model = MultinomialNaiveBayes()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate model
    metrics = evaluate_model(y_test, y_pred)
    
    return metrics

# Run Multinomial Naive Bayes on all datasets
results = {}
for dataset in ['enron1', 'enron2', 'enron4']:
    print(f"Running Multinomial Naive Bayes on {dataset}...")
    metrics = run_multinomial_nb(dataset)
    results[dataset] = metrics
    print(f"Metrics for {dataset}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    print()