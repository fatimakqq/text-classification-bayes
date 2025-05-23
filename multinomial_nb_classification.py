import numpy
import pandas
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class MultinomialNaiveBayes:
    def __init__(self):
        self.class_priors = None
        self.feature_probs = None
    
    def fit(self, X, y): #train the classifier
        # x is the feature matrix 
        # y is the label: spam vs ham

        num_samples, num_features = X.shape
        self.classes = numpy.unique(y)
        num_classes = len(self.classes)
        
        #calculate class priors
        self.class_priors = numpy.zeros(num_classes)
        for i, c in enumerate(self.classes):
            self.class_priors[i] = numpy.sum(y == c) / num_samples
        
        #calculate word probabilities
        #ADD ONE LAPLACE SMOOTHING
        self.feature_probs = numpy.zeros((num_classes, num_features))
        
        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            
            #total count of words for this class
            total_count = numpy.sum(X_c) + num_features  #add num_features for Laplace smoothing
            
            #calculate P(each word) with the smoothig 
            for j in range(num_features):
                word_count = numpy.sum(X_c[:, j]) + 1  #add 1
                self.feature_probs[i, j] = word_count / total_count
        
        #convert -> log probabilities
        self.log_class_priors = numpy.log(self.class_priors)
        self.log_feature_probs = numpy.log(self.feature_probs)
    
    def predict(self, X):
        #X is the feature matrix
        return numpy.array([self._predict_sample(x) for x in X])
    
    def _predict_sample(self, x): #single sample prediction

        #x is the feature vector
        posteriors = []
        
        #calc posterior probability for each class
        for i, c in enumerate(self.classes):
            
            log_posterior = self.log_class_priors[i]
            #add log likelihood for each feature
            log_posterior += numpy.sum(x * self.log_feature_probs[i]) #multiplication bc BoW
            posteriors.append(log_posterior)
        
        #return the class with the highest posterior prob
        return self.classes[numpy.argmax(posteriors)]

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
        'f1': f1_score(y_true, y_pred)}
    return metrics

def run_multinomial_nb(dataset_name):

    #create files and load the data - ONLY BAG OF WORDS
    train_csv = f"{dataset_name}_bow_train.csv"
    X_train, y_train = load_data(train_csv)

    test_csv = f"{dataset_name}_bow_test.csv"
    X_test, y_test = load_data(test_csv)
    
    #train based on our model
    model = MultinomialNaiveBayes()
    model.fit(X_train, y_train)
    
    y_prediction = model.predict(X_test)
    
    #run evals
    metrics = evaluate_model(y_test, y_prediction)
    return metrics

#run MNB on all 3 datasets
results = {}
for dataset in ['enron1', 'enron2', 'enron4']:
    metrics = run_multinomial_nb(dataset)
    results[dataset] = metrics
    print(f"\ncurrent dataset: {dataset}")
    for metric, value in metrics.items():
        print(f"{metric} = {value}")
