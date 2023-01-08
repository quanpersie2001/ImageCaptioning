import pickle
from tqdm import tqdm
from rouge import Rouge
from utils import k_beam_search
from nltk.translate.bleu_score import corpus_bleu


def calculate_rouge_score(actual, predict):
    actual_cap, pred_cap = list(), list()
    for tmp_1 in actual:
        for tmp_2 in tmp_1:
          actual_cap.append(tmp_2)

    for i in range(len(predict)):
        temp = []
        for j in range(5):
            temp.append(predict[i])
        pred_cap = pred_cap + temp

    rouge = Rouge()
    score = rouge.get_scores(hyps = pred_cap, refs = actual_cap, avg=True)
    return score


def evaluate_model(model, captions, photos_fe, k_beam= 3, log=False, verbose=True, mode='single'):
    """
    Calculate BLEU score of predictions
    """
    actual, predicted = [], []
    actual_r, predicted_r = [], []

    with open('process_data/word_to_id.pkl','rb') as f:
        word_to_id = pickle.load(f)
    with open('process_data/id_to_word.pkl','rb') as f:
        id_to_word = pickle.load(f)
    with open('process_data/max_length.pkl','rb') as f:
        max_length = pickle.load(f)

    # step over the whole set
    for key, desc_list in tqdm(captions.items()):
        yhat = k_beam_search(model, photos_fe[key], word_to_id, id_to_word, max_length, k_beam, log, mode)
        
        # store actual and predicted
        references = [d.split() for d in desc_list[:5]]
        actual.append(references)
        predicted.append(yhat.split())

        actual_r.append([d for d in desc_list[:5]])
        predicted_r.append(yhat)

    # calculate Rouge score
    score = calculate_rouge_score(actual_r, predicted_r)

    # calculate BLEU score
    b1=corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
    b2=corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
    b3=corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0))
    b4=corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))
    if verbose:
        print('\n')
        for i, b in enumerate([b1,b2,b3,b4], 1):
            print(f'BLEU-{i}: {b}')
        print('ROUGE: ', score)
    return [b1, b2, b3, b4, score]