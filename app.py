import warnings
import logging
import gradio as gr
import requests
import yfinance as yf
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import matplotlib.pyplot as plt

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gradio_client")
warnings.filterwarnings("ignore", category=FutureWarning)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load the FinBERT model and tokenizer
def load_finbert():
    try:
        logger.info("Loading FinBERT model and tokenizer...")
        tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
        model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
        logger.info("FinBERT model and tokenizer loaded successfully.")
        return tokenizer, model
    except Exception as e:
        logger.error(f"Error loading FinBERT model: {e}")
        raise

# Initialize the model and tokenizer
tokenizer, model = load_finbert()

# Analyze sentiment using FinBERT
def analyze_sentiment_finbert(text):
    if not text:  # Handle empty or None text
        return "neutral", 0.0  # Default to neutral if text is empty
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        sentiment_score = predictions[0].detach().numpy()
        sentiment_labels = ["negative", "neutral", "positive"]
        return sentiment_labels[sentiment_score.argmax()], sentiment_score.max()
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        return "error", 0.0

# Function to get stock data using Yahoo Finance
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1mo")
        # Convert PeriodIndex to DatetimeIndex to avoid warnings
        data.index = pd.to_datetime(data.index)
        return data
    except Exception as e:
        logger.error(f"Error fetching stock data for ticker '{ticker}': {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error

# Generate a stock price chart
def get_stock_chart(stock_data):
    if stock_data.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data available", horizontalalignment="center", verticalalignment="center")
        return fig

    fig, ax = plt.subplots()
    stock_data['Close'].plot(ax=ax, title="Stock Price Data", color="blue")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    plt.xticks(rotation=45)
    return fig

# Fetch news data from News API
def get_news(api_key, query):
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)
        return response.json().get('articles', [])
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching news for query '{query}': {e}")
        return []

# Main function to fetch stock data, news articles, and analyze sentiment
def financial_news_sentiment(api_key, ticker):
    # Fetch stock data and create stock chart
    stock_data = get_stock_data(ticker)
    stock_chart = get_stock_chart(stock_data)

    # Fetch and process news articles
    news_articles = get_news(api_key, ticker)
    if not news_articles:
        logger.warning("No news articles found.")
    news_summaries = []
    for article in news_articles:
        title = article.get('title', "No Title")
        summary = article.get('description', "")  # Use empty string if None

        # Use summary if available, else use title for sentiment analysis
        text_to_analyze = summary if summary else title
        sentiment_label, sentiment_score = analyze_sentiment_finbert(text_to_analyze)

        published_date = article.get('publishedAt', "Unknown Date")  # Default date
        news_summaries.append({
            "Title": title,
            "Summary": summary if summary else "Used Title for Sentiment",  # Indicate if title was used
            "Sentiment": sentiment_label,
            "Sentiment Score": sentiment_score,
            "Date": published_date
        })

    # Create a DataFrame of news summaries and sentiment, sorted by date
    news_df = pd.DataFrame(news_summaries)
    if not news_df.empty:
        news_df['Date'] = pd.to_datetime(news_df['Date'], errors='coerce')  # Convert Date column to datetime
        news_df = news_df.sort_values(by='Date', ascending=False)  # Sort by date

    # Calculate sentiment counts and proportions
    sentiment_counts = news_df['Sentiment'].value_counts()
    total_count = sentiment_counts.sum()
    sentiment_counts_dict = sentiment_counts.to_dict()  # Numeric counts for gr.Label
    sentiment_proportions = {label: f"{count} ({count / total_count * 100:.1f}%)"
                             for label, count in sentiment_counts.items()}

    # Format sentiment proportions as a displayable string
    formatted_proportions = "\n".join([f"{label}: {proportion}" for label, proportion in sentiment_proportions.items()])

    return stock_chart, news_df, sentiment_counts_dict, formatted_proportions

# Set up Gradio interface
interface = gr.Interface(
    fn=financial_news_sentiment,
    inputs=[gr.Textbox(label="Enter your News API Key"), gr.Textbox(label="Enter Stock Ticker")],
    outputs=[gr.Plot(label="Stock Price Chart"),
             gr.Dataframe(label="News and Sentiment Analysis"),
             gr.Label(label="Sentiment Counts"),
             gr.Textbox(label="Sentiment Counts with Proportions")]
)

# Launch the Gradio app
if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)
