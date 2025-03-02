import numpy
import pandas
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class DiscreteNaiveBayes:
    def __init__(self):
        self.class_priors = None
        self.feature_probs = None
    
    def fit(self, X, y): #train the DNB
        #x is feature matrix (bernoulli)
        #y is label: spam vs ham
        num_samples, num_features = X.shape
        self.classes = numpy.unique(y)
        num_classes = len(self.classes)
        
        self.class_priors = numpy.zeros(num_classes)
        for i, c in enumerate(self.classes):
            self.class_priors[i] = numpy.sum(y == c) / num_samples
        
        self.feature_probs = numpy.zeros((num_classes, num_features))
        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            n_docs = X_c.shape[0]
            
            #laplace smoothing
            for j in range(num_features):
                word_present = numpy.sum(X_c[:, j]) + 1
                self.feature_probs[i, j] = word_present / (n_docs + 2) 
        
        #use log space
        self.log_class_priors = numpy.log(self.class_priors)
        self.log_feature_probs = numpy.log(self.feature_probs)
        self.log_feature_absent_probs = numpy.log(1 - self.feature_probs)
    
    def predict(self, X):
        #X = feature matrix
        predictions = []
        for x in X:
            predicted_label = self._predict_sample(x)
            predictions.append(predicted_label)

        predicted_y = numpy.array(predictions)
        return predicted_y
    
    def _predict_sample(self, x):
        #single sample prediction
        posteriors = []
        
        for i, c in enumerate(self.classes):
            #log prior
            log_posterior = self.log_class_priors[i]
            
            for j in range(len(x)):
                if x[j] == 1:
                    log_posterior += self.log_feature_probs[i, j]
                else:
                    log_posterior += self.log_feature_absent_probs[i, j]
            posteriors.append(log_posterior)
        return self.classes[numpy.argmax(posteriors)] #highest posterior prob

def load_data(csv_file):
    data = pandas.read_csv(csv_file)
    y = data['label'].values
    X = data.drop('label', axis=1).values
    return X, y

def evaluate_model(y_true, y_pred):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }
    return metrics

def run_discrete_nb(dataset_name): 
    #BERNOULLI only
    train_file = f"{dataset_name}_bernoulli_train.csv"
    test_file = f"{dataset_name}_bernoulli_test.csv"
    
    X_train, y_train = load_data(train_file)
    X_test, y_test = load_data(test_file)
    
    model = DiscreteNaiveBayes()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    metrics = evaluate_model(y_test, y_pred)
    
    return metrics

#run DNB on all 3 datasets
results = {}
for dataset in ['enron1', 'enron2', 'enron4']:
    metrics = run_discrete_nb(dataset)
    results[dataset] = metrics
    print(f"\ncurrent dataset: {dataset}")
    for metric, value in metrics.items():
        print(f"{metric} = {value}")