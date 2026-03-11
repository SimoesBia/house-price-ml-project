import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# carregar dataset
housing = fetch_california_housing()

# transformar em tabela
df = pd.DataFrame(housing.data, columns=housing.feature_names)

# adicionar coluna de preço
df["price"] = housing.target

print(df.head())
print(df.info())

plt.figure()
sns.histplot(df["price"], bins=30)
plt.title("Distribuição dos preços das casas")
plt.show()

print(df.describe())

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlação entre variáveis")
plt.show()

plt.figure()
sns.scatterplot(x=df["MedInc"], y=df["price"])
plt.title("Renda média vs Preço das casas")
plt.xlabel("Renda média da região")
plt.ylabel("Preço da casa")
plt.show()

# separar dados de entrada e saída
X = df.drop("price", axis=1)
y = df["price"]

# dividir dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# criar modelo
model = LinearRegression()
# treinar modelo
model.fit(X_train, y_train)
# fazer previsões
predictions = model.predict(X_test)

print("Algumas previsões do modelo:")
print(predictions[:10])

# calcular métricas
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("MAE:", mae)
print("MSE:", mse)
print("R2:", r2)

plt.figure()
sns.scatterplot(x=y_test, y=predictions)

plt.xlabel("Preço real")
plt.ylabel("Preço previsto")
plt.title("Previsão vs Valor real")

plt.show()


# criar modelo Random Forest
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

print("Random Forest MAE:", rf_mae)
print("Random Forest R2:", rf_r2)

# obter importância das variáveis
importances = rf_model.feature_importances_

# criar tabela com resultados
feature_importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": importances
})

# ordenar da mais importante para a menos importante
feature_importance_df = feature_importance_df.sort_values(by="importance", ascending=False)

print(feature_importance_df)

plt.figure(figsize=(10,6))

sns.barplot(
    x="importance",
    y="feature",
    data=feature_importance_df
)

plt.title("Importância das variáveis na previsão de preços")
plt.xlabel("Importância")
plt.ylabel("Variável")

plt.show()

scores = cross_val_score(
    rf_model,
    X,
    y,
    cv=5,
    scoring="r2"
)

print("Cross-validation R2 scores:", scores)
print("Average R2:", scores.mean())

param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [None, 10, 20]
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=3,
    scoring="r2"
)

grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)

joblib.dump(rf_model, "house_price_model.pkl")

print("Modelo salvo com sucesso!")