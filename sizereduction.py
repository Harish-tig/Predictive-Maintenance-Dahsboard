# This code reduces the size of a dataset by randomly sampling 10% of the rows and saving it as a new CSV file.
import pandas as pd

# Load dataset
df = pd.read_csv(r"E:\Major Project\Dataset 3\Original Data\0Nm_misalignment_vibration\0Nm_Misalign_01.csv")

# Randomly sample 10% of rows
reduced_df = df.sample(frac=0.1, random_state=42)

# Save reduced dataset
reduced_df.to_csv(r"E:\Major Project\Dataset 3\Original Data\0Nm_misalignment_vibration\Red_0Nm_Misalign_01.csv", index=False)

print("Reduced dataset created with 10% of original rows.")