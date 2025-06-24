import pandas as pd
from TradeModel import TradeModel 
import matplotlib.pyplot as plt
#from sklearn.metrics import mean_absolute_error

target_date = "2025-06-09"
path = "/Users/jingyuanhe/code/algotrading/data/"
# Load data
data_df = pd.read_parquet(path + f"dataset_{target_date}.parquet")
data_df = data_df.drop('gold', axis=1)
data_df = data_df.drop('bond', axis=1)
train_ration = 0.8

cut_off = int(train_ration * data_df.shape[0])

training_df = data_df[:cut_off]
testing_df = data_df[cut_off:]
#print(training_df)
Y = training_df.iloc[:, 0]
X = training_df.iloc[:, 1:]
X = X.to_numpy()
X = X.reshape((X.shape[0], 5, int(X.shape[1] / 5)))
print(X.shape)
model = TradeModel()
model.create_model(X)
model.model_fit(X, Y)

Y_test = testing_df.iloc[:, 0]
X_test = testing_df.iloc[:, 1:]
X_test = X_test.to_numpy()
X_test = X_test.reshape((X_test.shape[0], 5, int(X_test.shape[1] / 5)))
predict = model.predict_next_price(X_test)

#mae = mean_absolute_error(Y_test, predic)
#print("Mean Absolute Error (MAE):", mae)
print(predict)
print(Y_test)
plt.figure(figsize=(8, 5))
plt.plot(Y_test, label='Actual', marker='o')
plt.plot(predict, label='Predicted', marker='x')
plt.title(f'Actual vs Predicted Values)')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(path + '../prediction_vs_actual.png', dpi=300)
#plt.show()
