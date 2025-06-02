# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 09:29:39 2025

@author: Allen Jones
"""

"""
Exercise 4: Text Analysis of Student Feedback
Problem:
An online learning platform collects textual feedback from students after each 
course. The product team wants to identify common themes and sentiment trends 
to improve course content and instructor performance. They are particularly 
interested in finding specific areas of improvement.
Instructions:
Use Python's natural language processing capabilities to analyze the textual 
feedback data. Implement sentiment analysis to gauge overall satisfaction, and 
use topic modeling to identify common themes in the feedback. Create 
visualizations to present your findings in a way that would be useful for 
course designers.
Solution:

"""

import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.data.path.append("C:/Users/ajones/AppData/Roaming/nltk_data")
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
#import pyLDAvis
#import pyLDAvis.sklearn
from textblob import TextBlob
import re
from wordcloud import WordCloud
from collections import Counter

# Download NLTK resources (run once)
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')

# Load the dataset and convert submission_date
feedback_data = pd.read_csv('student_feedback.csv')
feedback_data['submission_date'] = pd.to_datetime(feedback_data['submission_date'])

# Identify all feedback category titles
unique_categories = feedback_data['feedback_category'].unique()
print(f"Found category titles: {unique_categories}")



"""

# --- This is where the main new structure begins ---
# We'll store results for each category, maybe in a list or dictionary
all_category_results = [] # Let's store our processed DataFrames here

for category_name in unique_categories:
    print(f"\n--- Processing category: {category_name} ---")

    # 1. Filter data for the current category
    # We use .copy() to ensure we're working with a new DataFrame slice
    category_specific_data = feedback_data[feedback_data['feedback_category'] == category_name].copy()

    # Check if there's any data for this category
    if category_specific_data.empty:
        print(f"No data found for category: {category_name}")
        continue # Skip to the next category

    # 2. Apply preprocessing to the 'feedback_text' of this specific category
    print("Preprocessing text...")
    category_specific_data['processed_feedback'] = category_specific_data['feedback_text'].apply(preprocess_text)

    # 3. Sentiment Analysis for this category's original 'feedback_text'
    print("Performing sentiment analysis...")
    category_specific_data['sentiment'] = category_specific_data['feedback_text'].apply(
        lambda x: TextBlob(x).sentiment.polarity if isinstance(x, str) else 0
    )

    # 4. Categorize sentiment for this category
    category_specific_data['sentiment_category'] = pd.cut(
        category_specific_data['sentiment'],
        bins=[-1, -0.2, 0.2, 1],
        labels=['Negative', 'Neutral', 'Positive'],
        include_lowest=True # Good practice to include the -1
    )

    # At this point, category_specific_data contains the processed text and sentiment for one category
    # You can print some info to check:
    print(f"Sample of processed data for '{category_name}':")
    print(category_specific_data[['feedback_text', 'processed_feedback', 'sentiment', 'sentiment_category']].head())

    all_category_results.append(category_specific_data)

    # --- Placeholder for category-specific topic modeling & visualizations ---
    # We will integrate topic modeling and visualizations for THIS category_specific_data here later.
    # For now, the old topic modeling and visualization code below this loop will likely not work as intended
    # or will need to be removed/adapted.

# --- End of the new loop ---

# At this point, 'all_category_results' is a list of DataFrames,
# one for each feedback category, with its own processed text and sentiment.

# The OLD global analysis below this point will need to be re-evaluated.
# For example, the original:
# feedback_data['processed_feedback'] = feedback_data['feedback_text'].apply(preprocess_text)
# feedback_data['sentiment'] = feedback_data['feedback_text'].apply(...)
# feedback_data['sentiment_category'] = pd.cut(...)
# ... and the plots that rely on these global columns, will need to be adapted or removed.

"""











