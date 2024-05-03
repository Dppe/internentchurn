import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sb
import joblib

df = pd.read_csv("D:\PycharmProjects\MachineLearning\Internet_user_churn_predictor\internet_service_churn.csv")
ndf = df.fillna(0)
x = ndf.drop(["churn","id"], axis=1)
y = ndf["churn"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

plt.scatter(x["service_failure_count"],y,cmap="greens")
plt.xlabel("features")
plt.ylabel("Churn_Rate")
plt.show()

model = RandomForestClassifier(n_estimators=200, criterion="entropy")

print("--------------------------")
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Score Stacking :", accuracy * 100, "%")

x_train.to_csv("D:\PycharmProjects\MachineLearning\Internet_user_churn_predictor\X_train.csv", index=False)
x_test.to_csv("D:\PycharmProjects\MachineLearning\Internet_user_churn_predictor\X_test.csv", index=False)
pd.DataFrame(y_train, columns=["churn"]).to_csv("D:\PycharmProjects\MachineLearning\Internet_user_churn_predictor\y_train.csv", index=False)
pd.DataFrame(y_test, columns=["churn"]).to_csv("D:\PycharmProjects\MachineLearning\Internet_user_churn_predictor\y_test.csv", index=False)

pd.DataFrame(y_pred, columns=["PredictedChurn"]).to_csv("D:\PycharmProjects\MachineLearning\Internet_user_churn_predictor\predicted_Churn.csv", index=False)
joblib.dump(model, "D:\PycharmProjects\MachineLearning\Internet_user_churn_predictor\Churn_Rate_prediction_model.pkl")