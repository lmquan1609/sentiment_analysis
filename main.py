import torch
import torch.nn as nn
import pickle
import re

from simple_tokenizer import SimpleTokenizer

EMBEDDING_SIZE = 64
MAX_LENGTH = 32

def clean_sentence(string):
    strip_special_chars = re.compile("[^\w0-9 ]+")
    return re.sub(strip_special_chars, '', string.lower())

def load_tokenizer(tokenizer_path: str) -> SimpleTokenizer:
    """Load the tokenizer from the input path

    Args:
        tokenizer_path (str): [description]
    
    Returns:
        SimpleTokenizer
    """    
    with open(tokenizer_path, 'rb') as f:
        data = pickle.load(f)
        token2idx = data['token2idx']
        idx2token = data['idx2token']
        pad = data['pad']
        unknown = data['unknown']
        dismissed_tokens = data['dismissed_tokens']

        tokenizer = SimpleTokenizer(token2idx, idx2token, pad, unknown)
    return tokenizer

def load_model(model_path: str, tokenizer: SimpleTokenizer):
    """Load the model from model path

    Args:
        model_path (str): path to the model
        tokenizer (SimpleTokenizer): the object of SimpleTokenizer
    """ 
    VOCAB_SIZE = len(tokenizer.token2idx)

    model = nn.Sequential(
        nn.Embedding(VOCAB_SIZE, EMBEDDING_SIZE, padding_idx=tokenizer.pad),
        nn.Flatten(),
        nn.Linear(EMBEDDING_SIZE * MAX_LENGTH, 10),
        nn.ReLU(),
        nn.Linear(10, 1),
        nn.Sigmoid()
    )

    model.load_state_dict(torch.load(model_path))
    return model

def predict(sentence: str, model: nn.Sequential, tokenizer: SimpleTokenizer, device):
    sentence_cleaned = clean_sentence(sentence)
    sample_sequence = tokenizer.texts_to_sequences([sentence_cleaned], MAX_LENGTH)
    sample_sequence = torch.LongTensor(sample_sequence)

    preds = model(sample_sequence)

    if preds.detach().cpu().numpy()[0][0] > 0.5:
        return True
    else:
        return False

def main():
    tokenizer = load_tokenizer('./tokenizer.pickle')
    model = load_model('./model.pt', tokenizer)
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    while True:
        sentence = input('Input your sentence: ')
        result = predict(sentence, model, tokenizer, DEVICE)
        if result:
            print('It is Positive')
        else:
            print('It is Negative')

if __name__ == '__main__':
    main()