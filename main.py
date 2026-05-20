from dataloader import load_normalized_data

class KNN():
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self
    
    def _compute_distance(self, a, b):
        squared_sum = 0
        for index in range(len(a)):
            squared_sum += (a[index] - b[index]) ** 2
        return squared_sum ** 0.5

    def _predict_one(self, x):
        distances = []
        for index in range(len(self.X_train)):
            distance = self._compute_distance(x, self.X_train[index])
            label = self.y_train.iloc[index]
            distances.append((distance, label))
        
        distances.sort(key=lambda pair: pair[0])
        
        votes = {}
        for index in range(self.k):
            label = distances[index][1]
            if label not in votes:
                votes[label] = 0
            votes[label] += 1
        
        best_label = None
        best_count = 0
        for label in votes:
            if votes[label] > best_count:
                best_count = votes[label]
                best_label = label
        return best_label
    
    def predict(self, X):
        predictions = []
        for x in X:
            prediction = self._predict_one(x)
            predictions.append(prediction)
        return predictions
    
    def evaluate(self, X, y):
        predictions = self.predict(X)
        correct = 0
        for index in range(len(predictions)):
            if predictions[index] == y.iloc[index]:
                correct += 1
        accuracy = correct / len(predictions)
        return accuracy
    
    def grid_search(self, X_train, y_train, X_test, y_test, k_values):
        best_k = None
        best_accuracy = 0
        
        for k in k_values:
            self.k = k
            self.fit(X_train, y_train)
            accuracy = self.evaluate(X_test, y_test)
            print(f"k={k}, accuracy={accuracy}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_k = k
        
        self.k = best_k
        print(f"Meilleur k: {best_k} avec accuracy: {best_accuracy}")
        return best_k, best_accuracy
    
    
if __name__ == "__main__":
    X, y, scaler = load_normalized_data(file_path="data/Bien-être.csv")

    train_size = int(len(X) * 0.8)

    X_train = X[:train_size]
    y_train = y[:train_size]

    X_test = X[train_size:]
    y_test = y[train_size:]

    mon_knn = KNN()

    k_values = [1, 3, 5, 7, 9]
    meilleur_k, meilleure_accuracy = mon_knn.grid_search(
        X_train, y_train,
        X_test, y_test,
        k_values
    )

    print("\n=== RÉSULTAT FINAL ===")
    print(f"Meilleur k : {meilleur_k}")
    print(f"Accuracy : {meilleure_accuracy:.2f}")