from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv("final.csv")

mass_list = df["Mass_Star"].tolist()
mass_list.pop(0)

radius_list = df["Radius_Star"].tolist()
radius_list.pop(0)

X = []
for index, planet_mass in enumerate(mass_list):
    temp_list = [
        radius_list[index],
        planet_mass
    ]
    X.append(temp_list)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans_.inertia)

plt.figure()
sns.lineplot(range(1, 11), wcss, marker="o", color="red")
plt.title("Star Mass and Radius")
plt.xlabel("Mass")
plt.ylabel("Radius")
plt.show()
