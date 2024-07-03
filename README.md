# Sentixtract

Sentixtract is a Python repository designed for scraping articles from URLs and performing Natural Language Processing (NLP) techniques, focusing primarily on sentiment analysis. This tool extracts textual content from web articles, cleans and processes the text data, and computes sentiment scores and other linguistic metrics to derive insights from the extracted content.

## Features

- **Article Extraction**: Automatically fetches articles from specified URLs and saves them locally.
- **Text Preprocessing**: Cleans and tokenizes text data, removing non-alphabetic characters and stop words.
- **Sentiment Analysis**: Calculates sentiment scores based on the presence of positive and negative words.
- **Readability Metrics**: Computes readability scores such as average sentence length and Fog Index.
- **Additional Linguistic Metrics**: Calculates syllable counts, personal pronouns, and other linguistic features.
- **Progress Tracking**: Utilizes tqdm to display progress bars for tasks like data extraction and analysis.

## Dependencies

- Python (3.11.3 recommended)
- pandas (2.0.3)
- requests (2.31.0)
- tqdm (4.65.0)
- beautifulsoup4 (4.12.2)
- nltk (3.8.1)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/Sentixtract.git
   cd Sentixtract

2. Install dependencies:

   ```bash
   pip install -r requirements.txt

## Usage
1. Prepare your input data in an Excel file (`input_data.xlsx') containing columns 'URL' and 'URL_ID'
2. Run the main script to extract articles and analyze them:
   ```bash
   python sentixtract.py input_data.xlsx --article_dir articles --master_dict_dir MasterDictionary --output_file Output.xlsx
  Replace input_data.xlsx with your input file path and adjust directory paths as necessary.
3. Review the generated output in Output.xlsx for sentiment scores, readability metrics, and other analyzed features.
