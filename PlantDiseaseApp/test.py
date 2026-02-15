import os

print("Real:", len(os.listdir("training_data/real")))
print("Fake:", len(os.listdir("training_data/fake")))
