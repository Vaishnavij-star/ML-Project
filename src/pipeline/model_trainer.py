from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

class ModelTrainer:
    def initiate_model_training(self, X_train, X_test, y_train, y_test):

        model = LinearRegression()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        score = r2_score(y_test, preds)

        return model, score
