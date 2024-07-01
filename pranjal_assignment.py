import os
import re
import argparse
import requests
from tqdm import tqdm
import pandas as pd
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize, sent_tokenize
#nltk.download('punkt')
 
def load_stopwords() -> set:
    """
    Loads a set of stop words from a directory containing text files.

    Parameters:
    - No parameters are required.

    Returns:
    - A set of stop words.
    
    Raises:
    - This function does not raise any exceptions.
    
    Usage:
    >>> stop_words = load_stopwords()
    """
    stopwords = set()
    
    for file_name in os.listdir("StopWords"):
        with open(os.path.join('StopWords', file_name), 'r', encoding='latin') as file:
            words = file.read().split()
            stopwords.update(words)

    return stopwords

stop_words = load_stopwords()

def load_master_dict(master_dict_dir):
    """
    Loads a master dictionary containing positive and negative words from a directory.

    Parameters:
    - master_dict_dir (str): The directory path where the master dictionary files are located.

    Returns:
    - positive_words (set): A set of positive words.
    - negative_words (set): A set of negative words.

    Raises:
    - FileNotFoundError: If the master dictionary files do not exist in the specified directory.
    - UnicodeDecodeError: If the master dictionary files contain non-UTF-8 encoded text.

    Usage:
    >>> positive_words, negative_words = load_master_dict('master_dicts')
    """
    positive_words = set()
    negative_words = set()

    try:
        with open(os.path.join(master_dict_dir, 'positive-words.txt'), 'r', encoding='utf-8') as file:
            positive_words.update(file.read().split())
    except UnicodeDecodeError:
        with open(os.path.join(master_dict_dir, 'positive-words.txt'), 'r', encoding='latin') as file:
            positive_words.update(file.read().split())

    try:  
        with open(os.path.join(master_dict_dir, 'negative-words.txt'), 'r', encoding='utf-8') as file:
            negative_words.update(file.read().split())
    except UnicodeDecodeError:
        with open(os.path.join(master_dict_dir, 'negative-words.txt'), 'r', encoding='latin') as file:
            negative_words.update(file.read().split())

    return positive_words, negative_words

def clean_and_tokenize(text, stop_words):
    """
    Cleans and tokenizes the input text by removing non-alphabetic characters,
    converting the text to lowercase, and removing stop words.

    Parameters:
    - text (str): The input text to be cleaned and tokenized.
    - stop_words (set): A set of stop words to be removed from the text.

    Returns:
    - cleaned_words (list): A list of cleaned and tokenized words.

    Raises:
    - This function does not raise any exceptions.

    Usage:
    >>> cleaned_words = clean_and_tokenize("This is a sample text.", stop_words)
    """
    words = word_tokenize(text.lower())
    cleaned_words = [word for word in words if word.isalpha() and word not in stop_words]
    return cleaned_words

def calculate_sentiment_scores(words, positive_words, negative_words):
    """
    Calculates the sentiment scores of a given list of words based on the presence of positive and negative words.
   
    Parameters:
    - words (list): A list of words to be analyzed for sentiment.
    - positive_words (set): A set of positive words to be considered in the sentiment analysis.
    - negative_words (set): A set of negative words to be considered in the sentiment analysis.
    
    Returns:
    - positive_score (int): The number of positive words found in the input list.
    - negative_score (int): The number of negative words found in the input list.
    - polarity_score (float): A score between -1 and 1 representing the overall sentiment polarity of the input list.
    - subjectivity_score (float): A score between 0 and 1 representing the overall subjectivity of the input list.
    """
    positive_score = sum(1 for word in words if word in positive_words)
    negative_score = sum(1 for word in words if word in negative_words)
    
    polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)
    subjectivity_score = (positive_score + negative_score) / (len(words) + 0.000001)
    
    return positive_score, negative_score, polarity_score, subjectivity_score

def calculate_readability_scores(text: str) -> tuple:
    """
    Calculates the readability scores of a given text.

    Parameters:
    - text (str): The input text to be analyzed for readability.

    Returns:
    - tuple: A tuple containing the average sentence length, percentage of complex words, Fog Index, number of complex words, and total number of words.
    
    Definitions:
    - The Fog Index is a measure of the readability of a given text. It is calculated using the following formula:
    - Fog Index = 0.4 * (Average Sentence Length + Percentage of Complex Words)
    - The average sentence length is calculated by dividing the total number of words by the number of sentences. The percentage of complex words is calculated by dividing the number of complex words by the total number of words.
    - Complex words are words with more than two vowels.
    """
    sentences = sent_tokenize(text)
    words = word_tokenize(text)

    num_sentences = len(sentences)
    num_words = len(words)
    avg_sentence_length = num_words / num_sentences if num_sentences > 0 else 0

    complex_words = [word for word in words if len([char for char in word if char in 'aeiou']) > 2]
    percentage_complex_words = len(complex_words) / num_words if num_words > 0 else 0

    fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)

    return avg_sentence_length, percentage_complex_words, fog_index, len(complex_words), num_words

def calculate_additional_scores(words):
    
    """
    Calculates additional scores for a given list of words.

    Parameters:
    - words (list): A list of words to be analyzed for additional scores.

    Returns:
    - syllable_count (int): The total number of syllables in the input list of words.
    - personal_pronouns (int): The number of personal pronouns (I, we, my, ours, us) found in the input list of words.
    - avg_word_length (float): The average length of words in the input list of words.
    - avg_syllables_per_word (float): The average number of syllables per word in the input list of words.

    This function calculates the total number of syllables, the number of personal pronouns, the average word length, and the average number of syllables per word in a given list of words. The total number of syllables is calculated by counting the number of vowels in each word. Personal pronouns are counted by searching for the words "I", "we", "my", "ours", and "us" in the input list of words. The average word length is calculated by dividing the total number of characters in all words by the total number of words. The average number of syllables per word is calculated by dividing the total number of syllables by the total number of words.
    """
    syllable_count = sum([len(re.findall(r'[aeiou]', word)) for word in words])
    personal_pronouns = len(re.findall(r'\b(I|we|my|ours|us)\b', ' '.join(words), re.I))
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
    avg_syllables_per_word = syllable_count / len(words) if words else 0

    return syllable_count, personal_pronouns, avg_word_length, avg_syllables_per_word

def extract_data(input_file, output_dir='articles'):
    """
    Extracts data from a given input file and saves it to the specified output directory.

    Parameters:
    - input_file (str): The path to the input Excel file containing URLs and URL IDs.
    - output_dir (str, optional): The directory where the extracted data will be saved. Default is 'articles'.

    Returns:
    - None. The function saves the extracted data to the specified output directory.

    Raises:
    - FileNotFoundError: If the input file does not exist.
    - NotADirectoryError: If the output directory does not exist.

    Usage:
    >>> extract_data('input_data.xlsx', 'output_directory')
    """
    df = pd.read_excel(input_file)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Wrap the loop with tqdm for progress tracking
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Extracting data"):
        url = row['URL']
        url_id = row['URL_ID']

        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        title = soup.find('title').get_text()

        article_div = soup.find('div', class_='td-post-content tagdiv-type')
        if article_div:
            paragraphs = article_div.find_all(['p', 'ol', 'pre'])
            article_text = ' '.join([paragraph.get_text() for paragraph in paragraphs])

            with open(f"{output_dir}/{url_id}.txt", "w", encoding='utf-8') as file:
                file.write(f"{title}\n{article_text}")
        else:
            print(f"Article content not found for URL_ID {url_id}")

def analyze_data(article_dir, master_dict_dir, input_file, output_file="Output.xlsx"):
    """
    Analyzes the articles in the specified article directory and calculates various sentiment, readability, and additional scores.

    Parameters:
    - article_dir (str): The directory path where the articles are stored.
    - master_dict_dir (str): The directory path where the master dictionary files are located.
    - input_file (str): The path to the input Excel file containing URLs and URL IDs.
    - output_file (str, optional): The directory where the extracted data will be saved. Default is 'Output.xlsx'.

    Returns:
    - None. The function saves the extracted data to the specified output directory.

    Raises:
    - FileNotFoundError: If the input file does not exist.
    - NotADirectoryError: If the output directory does not exist.

    Usage:
    >>> analyze_data('articles', 'MasterDictionary', "Input.xlsx")
    """
    stop_words = load_stopwords()
    positive_words, negative_words = load_master_dict(master_dict_dir)

    input_df = pd.read_excel(input_file)
    url_mapping = dict(zip(input_df['URL_ID'], input_df['URL']))

    output_data = []

    file_list = [file_name for file_name in os.listdir(article_dir) if file_name.endswith(".txt")]

    for file_name in tqdm(file_list, desc="Processing articles"):
        url_id = file_name.split(".")[0]
        url = url_mapping.get(url_id, "")

        with open(os.path.join(article_dir, file_name), "r", encoding='utf-8') as file:
            title = file.readline().strip()
            article_text = file.read().strip()

        cleaned_words = clean_and_tokenize(article_text, stop_words)
        positive_score, negative_score, polarity_score, subjectivity_score = calculate_sentiment_scores(
            cleaned_words, positive_words, negative_words)
        avg_sentence_length, percentage_complex_words, fog_index, complex_wordcount, wordcount = calculate_readability_scores(
            article_text)
        syllable_count, personal_pronouns, avg_word_length, avg_syllables_per_word = calculate_additional_scores(
            cleaned_words)

        output_data.append([
            url_id, url, positive_score, negative_score, polarity_score, subjectivity_score,
            avg_sentence_length, percentage_complex_words, fog_index, avg_sentence_length,
            complex_wordcount, wordcount, avg_syllables_per_word, personal_pronouns, avg_word_length
        ])

    output_df = pd.DataFrame(output_data, columns=[
        'URL_ID', 'URL', 'POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE', 'SUBJECTIVITY SCORE',
        'AVG SENTENCE LENGTH', 'PERCENTAGE OF COMPLEX WORDS', 'FOG INDEX', 'AVG NUMBER OF WORDS PER SENTENCE',
        'COMPLEX WORD COUNT', 'WORD COUNT', 'SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH'
    ])

    output_df.to_excel(output_file, index=False)

def main():
    parser = argparse.ArgumentParser(description='Extract and analyze data from articles.')
    parser.add_argument('input_file', type=str, help='Path to the input Excel file containing URLs and URL IDs')
    parser.add_argument('--article_dir', type=str, default='articles', help='Directory where the extracted articles will be saved (default: articles)')
    parser.add_argument('--master_dict_dir', type=str, default='MasterDictionary', help='Directory where the master dictionary files are located (default: MasterDictionary)')
    parser.add_argument('--output_file', type=str, default='Output.xlsx', help='Path to the output Excel file (default: Output.xlsx)')

    args = parser.parse_args()

    # Call your functions with the arguments from argparse
    extract_data(args.input_file, args.article_dir)
    analyze_data(args.article_dir, args.master_dict_dir, args.input_file, args.output_file)
    
if __name__ == "__main__":
    main()