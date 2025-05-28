# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 21:38:52 2025

@author: AJones
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

# Load the dataset
feedback_data = pd.read_csv('student_feedback.csv')

# Preprocessing function
def preprocess_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        # Add custom words here
        custom_blocks = ['allen', 'class', 'student', 'course', 'students',
        'professor', 'jones', 'semester', 'dr']
        stop_words.update(custom_blocks)
        filtered_tokens = [word for word in tokens if word not in stop_words]
        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
        return ' '.join(lemmatized_tokens)
    else:
        return ''

# Apply preprocessing
feedback_data['processed_feedback'] = feedback_data['feedback_text'].apply(preprocess_text)

# Sentiment Analysis
feedback_data['sentiment'] = feedback_data['feedback_text'].apply(
    lambda x: TextBlob(x).sentiment.polarity if isinstance(x, str) else 0
)

# Categorize sentiment
feedback_data['sentiment_category'] = pd.cut(
    feedback_data['sentiment'], 
    bins=[-1, -0.2, 0.2, 1], 
    labels=['Negative', 'Neutral', 'Positive']
)

# Sentiment over time
plt.figure(figsize=(10, 6))
feedback_data['submission_date'] = pd.to_datetime(feedback_data['submission_date'])
feedback_data['month'] = feedback_data['submission_date'].dt.strftime('%Y-%m')
sentiment_by_month = feedback_data.groupby('month')['sentiment'].mean()
sentiment_by_month.plot(kind='line', marker='o')
plt.title('Average Sentiment Trend Over Time')
plt.xlabel('Month')
plt.ylabel('Average Sentiment (Polarity)')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Sentiment by course
plt.figure(figsize=(12, 6))
sentiment_by_course = feedback_data.groupby('course_id')['sentiment'].mean().sort_values()
sentiment_by_course.plot(kind='barh', color=plt.cm.RdYlGn(
    (sentiment_by_course.values + 1) / 2))  # Map -1,1 to 0,1 for colormap
plt.title('Average Sentiment by Course')
plt.xlabel('Average Sentiment (Polarity)')
plt.ylabel('Course ID')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

# Topic Modeling
vectorizer = CountVectorizer(max_features=1000, min_df=5)
X = vectorizer.fit_transform(feedback_data['processed_feedback'])
feature_names = vectorizer.get_feature_names_out()

# LDA model
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X)


# Print top words for each topic
def print_top_words(model, feature_names, n_top_words):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topics[f"Topic {topic_idx}"] = top_words
        print(f"Topic {topic_idx}:")
        print(" ".join(top_words))
    return topics


topics = print_top_words(lda, feature_names, 10)

# Assign topics to feedback
doc_topics = lda.transform(X)
feedback_data['primary_topic'] = pd.Series(doc_topics.argmax(axis=1))

# Visualize topic distribution
plt.figure(figsize=(10, 6))
topic_counts = feedback_data['primary_topic'].value_counts().sort_index()
sns.barplot(x=topic_counts.index, y=topic_counts.values)
plt.title('Distribution of Primary Topics in Student Feedback')
plt.xlabel('Topic Number')
plt.ylabel('Count')
plt.xticks(range(len(topics)), [f"Topic {i}" for i in range(len(topics))])
plt.tight_layout()
plt.show()

# Word cloud for each topic
for topic_idx, top_words in topics.items():
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(top_words))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {topic_idx}')
    plt.tight_layout()
    plt.show()

# Sentiment across topics
plt.figure(figsize=(10, 6))
topic_sentiment = feedback_data.groupby('primary_topic')['sentiment'].mean()
sns.barplot(x=topic_sentiment.index, y=topic_sentiment.values, 
            palette=plt.cm.RdYlGn((topic_sentiment.values + 1) / 2))
plt.title('Average Sentiment by Topic')
plt.xlabel('Topic')
plt.ylabel('Average Sentiment')
plt.xticks(range(len(topics)), [f"Topic {i}" for i in range(len(topics))])
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

# Areas for improvement (negative feedback analysis)
negative_feedback = feedback_data[feedback_data['sentiment_category'] == 'Negative']
negative_topics = negative_feedback['primary_topic'].value_counts()
print("Topics most commonly found in negative feedback:")
for topic, count in negative_topics.items():
    print(f"Topic {topic} ({count} mentions): {', '.join(topics[f'Topic {topic}'][:5])}")


