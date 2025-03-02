import numpy
import pandas
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iterations=300, lambda_reg=0.1):
        self.learning_rate = learning_rate #step size for gradient ascent
        self.max_iterations = max_iterations
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z): #sigmoid fc will map input to a val 0.0-1.0
        z = numpy.clip(z, -500, 500)  #hard limit
        return 1 / (1 + numpy.exp(-z))
    
    def fit(self, X, y):
        #gradient ascent
        num_samples, num_features = X.shape
        
        self.weights = numpy.zeros(num_features)#parameters
        self.bias = 0
        
        for _ in range(self.max_iterations):
            #go through it once
            linear_model = numpy.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)
            dw = (1/num_samples) * numpy.dot(X.T, (y - y_pred)) - (self.lambda_reg/num_samples) * self.weights
            db = (1/num_samples) * numpy.sum(y - y_pred)

            self.weights += self.learning_rate * dw #update
            self.bias += self.learning_rate * db
    
    def predict_prob(self, X):
        linear_model = numpy.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)
    
    def predict(self, X, threshold=0.5):
        y_prob = self.predict_prob(X)
        class_prediction  = [1 if i > threshold else 0 for i in y_prob]
        return class_prediction
    
def load_data(csv_file):
    data = pandas.read_csv(csv_file)
    y = data['label'].values
    X = data.drop('label', axis=1).values
    return X, y

def evaluate_model(y_true, y_pred):
    metrics_dict = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }
    return metrics_dict

def tune_lambda(dataset, representation): #find best lambda
    train_file = f"{dataset}_{representation}_train.csv"
    #load data
    X_train_full, y_train_full = load_data(train_file)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.3, random_state=42
    )
    
    
    lambda_values = [0.001, 0.01, 0.1, 1.0, 10.0] #testing  these vals
    results = {}
    
    print(f"tuning lambda for {representation} {dataset} ")
    
    #test each lambda value
    for lambda_val in lambda_values:
        #train
        model = LogisticRegression(learning_rate=0.1, max_iterations=300, lambda_reg=lambda_val)
        model.fit(X_train, y_train)
        
        #evaluate on validaiton set
        y_val_pred= model.predict(X_val)
        val_metrics = evaluate_model(y_val, y_val_pred)
        
        #store results
        results[lambda_val] = val_metrics
        print(f"lambda {lambda_val}: f1 = {val_metrics['f1']}, accuracy = {val_metrics['accuracy']}")
    
    #using f1 for best lambda
    best_lambda = max(results.items(), key=lambda x: x[1]['f1'])[0]
    print(f"best lambda: {best_lambda}")
    return best_lambda, results

def run_logistic_regression(dataset, representation):
    train_file = f"{dataset}_{representation}_train.csv"
    test_file = f"{dataset}_{representation}_test.csv"
    
    best_lambda, lambda_results = tune_lambda(dataset, representation)
    
    #load
    X_train, y_train = load_data(train_file)
    X_test, y_test = load_data(test_file)
    
    model = LogisticRegression(learning_rate=0.1, max_iterations=300, lambda_reg=best_lambda)
    model.fit(X_train, y_train)
    
    y_test_pred = model.predict(X_test)
    test_metrics = evaluate_model(y_test, y_test_pred)
    
    print(f"\n{dataset} {representation} representation:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value}")
    
    return best_lambda, test_metrics

#run fo all!
all_results = {}
for dataset in ['enron1', 'enron2', 'enron4']:
    dataset_results = {}
    for representation in ['bow', 'bernoulli']:
        best_lambda, metrics = run_logistic_regression(dataset, representation)
        dataset_results[representation] = {
            'best_lambda': best_lambda,
            'metrics': metrics
        }
    all_results[dataset] = dataset_results