import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Function to scrape article text from a given URL
def scrape_article_text(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract article text based on the actual HTML structure of the website
        article_text = ""
        paragraphs = soup.find_all('p')
        for paragraph in paragraphs:
            article_text += paragraph.text + '\n'
        return article_text.strip()
    else:
        return None

# Function to scrape news articles and tags from the BBC website
def scrape_bbc_articles(url, max_articles=100):
    articles = []
    tags = []

    # Send an HTTP request to the URL
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract news articles and their classifications
        article_elements = soup.find_all('div', class_='media__content')  # Adjust this based on the actual HTML structure

        for article in article_elements[:max_articles]:
            title = article.find('h3', class_='media__title').get_text()  # Adjust this based on the actual HTML structure
            classification = article.find('a', class_='media__tag').get_text()  # BBC does not explicitly provide classifications in the HTML

            #article_url = article.find('a')['href']
            #article_text = scrape_article_text(article_url)

            print(title)
            print(classification)

            #if article_text:
            articles.append(title)
            tags.append(classification)

    return articles, tags

# Define the URL of the BBC website
bbc_website_url = "https://www.bbc.com/"

# Scrape news articles and tags
articles, tags = scrape_bbc_articles(bbc_website_url)

# Create a DataFrame from the scraped data
df = pd.DataFrame({'Article': articles, 'Tag': tags})

print(df)

# Split the data into training and testing sets if the dataset is large enough
if len(df) > 1:
    X_train, X_test, y_train, y_test = train_test_split(df['Article'], df['Tag'], test_size=0.2, random_state=42)

    # Build a text classification model
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)

    # Save the model to a file (you can use joblib for this)
    model_filename = 'news_classification_model.joblib'
    # joblib.dump(model, model_filename)

    # Save the test evaluation to a CSV file
    evaluation_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    evaluation_df.to_csv('test_evaluation.csv', index=False)

    print(f"Model Accuracy: {accuracy}")
else:
    print("Not enough data to split into training and testing sets.")
