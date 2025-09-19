# Task 1: Load and Explore the Dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

def main():
    try:
        # Load Iris dataset
        iris = load_iris(as_frame=True)
        df = iris.frame  # Pandas DataFrame version
        df['species'] = df['target'].map(dict(enumerate(iris.target_names)))  # Map species names

        print("Dataset loaded successfully!\n")

        # Display first few rows
        print("First 5 rows of the dataset:")
        print(df.head(), "\n")

        # Data structure
        print("Dataset Info:")
        print(df.info(), "\n")

        # Check for missing values
        print("Missing values per column:")
        print(df.isnull().sum(), "\n")

        # Clean dataset (if missing values exist)
        if df.isnull().values.any():
            df = df.dropna()
            print("Missing values found and dropped.\n")
        else:
            print("No missing values detected.\n")
            
        # Task 2: Basic Data Analysis

        print("Basic Statistics:")
        print(df.describe(), "\n")

        print("Average values grouped by species:")
        group_means = df.groupby("species").mean(numeric_only=True)
        print(group_means, "\n")

        # Pattern finding
        print("Interesting Findings:")
        print("- Setosa species generally has the smallest petal length & width.")
        print("- Virginica has the largest overall flower dimensions.\n")
        
        # Task 3: Data Visualization

        # Line chart (trend example: cumulative petal length by index)
        plt.figure(figsize=(8,5))
        plt.plot(df.index, df['petal length (cm)'].cumsum(), label="Cumulative Petal Length", color='teal')
        plt.title("Line Chart: Cumulative Petal Length")
        plt.xlabel("Sample Index")
        plt.ylabel("Cumulative Petal Length (cm)")
        plt.legend()
        plt.show()

        # Bar chart (average petal length per species)
        plt.figure(figsize=(6,4))
        sns.barplot(x="species", y="petal length (cm)", data=df, palette="Set2", ci=None)
        plt.title("Bar Chart: Average Petal Length per Species")
        plt.xlabel("Species")
        plt.ylabel("Average Petal Length (cm)")
        plt.show()

        # Histogram (distribution of sepal length)
        plt.figure(figsize=(6,4))
        plt.hist(df['sepal length (cm)'], bins=15, color='skyblue', edgecolor='black')
        plt.title("Histogram: Distribution of Sepal Length")
        plt.xlabel("Sepal Length (cm)")
        plt.ylabel("Frequency")
        plt.show()

        # Scatter plot (sepal length vs petal length)
        plt.figure(figsize=(6,5))
        sns.scatterplot(x="sepal length (cm)", y="petal length (cm)", hue="species", data=df, palette="Set1")
        plt.title("Scatter Plot: Sepal Length vs Petal Length")
        plt.xlabel("Sepal Length (cm)")
        plt.ylabel("Petal Length (cm)")
        plt.legend(title="Species")
        plt.show()

    except FileNotFoundError:
        print("Error: Dataset file not found.")
    except pd.errors.EmptyDataError:
        print("Error: Dataset file is empty.")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
