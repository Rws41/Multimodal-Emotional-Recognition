import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def exploratory_analysis(data):
    df = data.copy()
    # Removing duplicates
    df.drop_duplicates(subset='Participant Id', inplace=True)
    y = np.arange(1,5)

    # Age distribution
    plt.bar(df['Age'].unique(),df['Age'].value_counts())
    plt.title("Age Frequency in YAAD participants")
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.yticks(y)
    plt.savefig('./data/Age_frequency.png')
    plt.close()

    # sex distribution
    plt.pie(df['Gender'].value_counts(), labels = df['Gender'].unique(), autopct='%1.1f%%')
    plt.title("Distribution of Male and Female in YAAD participants" )
    plt.savefig('./data/gender_distribution.png')
    plt.close()
    return