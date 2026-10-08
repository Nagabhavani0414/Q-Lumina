from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

print("===== Q-CURE CANCER DATASET =====")

print("Dataset shape:", data.data.shape)

print("\nFeature names:")
for i, name in enumerate(data.feature_names):
    print(i, ":", name)

print("\nTarget mapping:")
for i, name in enumerate(data.target_names):
    print(i, ":", name)

print("\nFirst sample - selected features:")

print("Radius   :", data.data[0][0])
print("Texture  :", data.data[0][1])
print("Perimeter:", data.data[0][2])
print("Area     :", data.data[0][3])

print("\nFirst sample target:", data.target[0])