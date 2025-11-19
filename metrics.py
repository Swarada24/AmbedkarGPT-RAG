from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer= True)

def compute_rouge_l(pred, truth):
    return scorer.score(truth, pred)["rougeL"].fmeasure

def compute_cosine(pred, truth):
    embeddings = embedder.encode([pred, truth])
    return float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])

def compute_bleu(pred, truth):
    return sentence_bleu([truth.split()], pred.split())

def compute_faithfulness(pred, context):
    pred_words = set(pred.lower().split())
    context_words = set(context.lower().split())
    overlap = pred_words & context_words
    return len(overlap) / max(len(pred_words), 1) 


def compute_relevance(pred, truth):
    return compute_cosine(pred, truth)