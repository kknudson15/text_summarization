import nltk
import heapq
import re

# Need to gather these resources once.
# nltk.download('stopwords')
# nltk.download('punkt')


def input_resource():
    file = input("What is the file that you would like to summarize?")
    return file


def output_resource():
    file = input("What is the file that you would like to output?")
    return file


def read_article(file_name):
    article = open(file_name, 'r')
    article_data = article.read()
    return article_data


def preprocess_text(text):
    text = re.sub(r'[^\w\s]','',text)
    return text


def sentence_tokenization(text):
    sentences = nltk.sent_tokenize(text)
    return sentences


def word_tokenization(text):
    words = nltk.word_tokenize(text)
    return words


def weighted_occurrence(text):
    stopwords = nltk.corpus.stopwords.words('english')

    word_frequencies = {}
    for word in nltk.word_tokenize(text):
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
    maximum_frequency = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word] / maximum_frequency)

    return word_frequencies


def calculate_sentence_scores(sentence_list, word_frequencies):
    sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]
    return sentence_scores


def display_summary(sentence_scores):
    summary_sentences = heapq.nlargest(2, sentence_scores, key=sentence_scores.get)

    summary = ' '.join(summary_sentences)
    return summary


def export_summary(filename, summary):
    f = open(filename, 'w+')
    f.write(summary)
    f.close()


def main():
    in_file = input_resource()
    text = read_article(in_file)
    formatted_text = preprocess_text(text)
    sentences = sentence_tokenization(text)
    word_frequencies = weighted_occurrence(formatted_text)
    sentence_scores = calculate_sentence_scores(sentences, word_frequencies)
    summary = display_summary(sentence_scores)
    file = output_resource()
    export_summary(filename=file, summary=summary)


if __name__ == "__main__":
    main()