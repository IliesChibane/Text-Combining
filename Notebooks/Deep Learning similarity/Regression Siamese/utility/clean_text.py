import re
import tensorflow as tf
import nltk
from nltk.corpus import stopwords


class VocabularyProcessor:
    def __init__(self, max_document_length=None, min_frequency=1):
        self.max_document_length = max_document_length
        self.min_frequency = min_frequency
        self.vocab_dict = {}
        self.vocab_list = []
        self.reverse_vocab_dict = {}

    def fit(self, documents):
        # Clean text 
        for sen in documents:
            self.clean_text(sen)
            self.remove_stopwords(sen)
        
        # Create a dataset from the input documents
        dataset = tf.data.Dataset.from_tensor_slices(documents)
        
        # Split the sentences into words
        dataset = dataset.map(lambda x: tf.strings.split(x))
        
        # Flatten the dataset to have a single list of words
        dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
        
        # Filter words based on their frequency
        if self.min_frequency > 0:
            # Count the frequency of each word
            counter = dataset.reduce(tf.lookup.KeyValueTensorInitializer([], []), lambda x, y: x.insert(y, x.lookup(y) + 1))
            
            # Filter words based on their frequency
            dataset = dataset.filter(lambda x: counter.lookup(x) >= self.min_frequency)
        
        # Create a vocabulary from the dataset
        vocab = tf.keras.layers.StringLookup()
        vocab.adapt(dataset)
        
        # Store the vocabulary for later use
        self.vocab_dict = {k: v-2 for v, k in enumerate(vocab.get_vocabulary()) if v > 0}
        self.vocab_list = sorted(self.vocab_dict, key=self.vocab_dict.get)
        self.reverse_vocab_dict = {v: k for k, v in self.vocab_dict.items()}

    def transform(self, documents):
        
        # Create a dataset from the input documents
        dataset = tf.data.Dataset.from_tensor_slices(documents)
        
        # Split the sentences into words
        dataset = dataset.map(lambda x: tf.strings.split(x))
        
        # Map words to their ids in the vocabulary        
        dataset = dataset.map(lambda x: tf.map_fn(lambda y: tf.py_function(lambda z: self.vocab_dict.get(z.numpy().decode('utf-8'), 0), [y], Tout=tf.int64), x, fn_output_signature=tf.int64))
        # Pad or truncate sentences to have a consistent length
        if self.max_document_length is not None:
            dataset = dataset.map(lambda x: tf.cond(tf.shape(x)[0] < self.max_document_length,
                                                    lambda: tf.concat([x, tf.zeros(self.max_document_length - tf.shape(x)[0], dtype=tf.int64)], axis=0),
                                                    lambda: x[:self.max_document_length]))
            
        
        return list(dataset.as_numpy_iterator())

    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)


    def clean_text(text):
        ''' Pre process and convert texts to a list of words '''
        text = str(text)
        text = text.lower()

        # Clean the text
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"\s{2,}", " ", text)


    def remove_stopwords(text):

        # Tokenize the text into words
        words = nltk.word_tokenize(text)

        # Get the list of English stop words
        stop_words = set(stopwords.words('english'))

        # Remove the stop words from the list of words
        filtered_words = [word for word in words if word.lower() not in stop_words]

        # Join the filtered words back into a single string
        text = ' '.join(filtered_words)


