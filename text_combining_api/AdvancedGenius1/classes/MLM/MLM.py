import transformers
from transformers import pipeline
from transformers import logging
logging.set_verbosity_error()

def init_distilbert():
    fill_mask = pipeline("fill-mask", model="distilbert-base-uncased")
    return fill_mask