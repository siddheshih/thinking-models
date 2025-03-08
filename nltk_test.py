# minimal_example.py
import nltk

# Check if the 'punkt_tab' resource is available, and download if it isn't
try:
    nltk.data.find('tokenizers/punkt_tab/english.pickle')
except LookupError:
    nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize

# # Your text processing code here
# text = "Your sample text here."
# tokens = word_tokenize(text)
# print(tokens)

# #Correct handling NLTK
# try:
#     nltk.data.find('corpora/stopwords')
# except LookupError:
#     nltk.download('stopwords')

# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt')

# text = "This is a test sentence with some numbers 123.45 and words."
# tokens = word_tokenize(text)
# print(tokens)