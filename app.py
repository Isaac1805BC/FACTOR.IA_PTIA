from flask import Flask, render_template, request, jsonify
import pickle
import os
import re
import time
import json

# ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np

app = Flask(__name__)

# ── Preprocessing (no NLTK needed) ──────────────────────────────────────────
STOP_WORDS = {
    'a','an','the','and','or','but','in','on','at','to','for','of','with',
    'by','is','was','are','were','be','been','being','have','has','had',
    'do','does','did','will','would','could','should','may','might','shall',
    'can','this','that','these','those','it','its','as','from','into','up',
    'out','about','after','before','over','under','then','than','so','not',
    'no','nor','if','when','where','who','which','what','how','he','she',
    'they','we','you','i','me','him','her','us','them','my','our','your',
    'his','their','its','all','any','both','each','few','more','most','other',
    'some','such','only','own','same','too','very','just','because','while',
    'also','well','back','down','get','put','set','let','still','even','said',
    'much','now','here','there','way','two','new','old','first','last','long',
    'great','little','own','right','big','high','different','small','large',
    'next','early','young','important','public','private','real','best','free',
    'through','during','without','within','between','against','however','although',
}

def simple_stem(word):
    """Minimal suffix stripping."""
    for suffix in ('ing', 'tion', 'ness', 'ment', 'edly', 'edly', 'edly',
                   'ers', 'er', 'ed', 'ly', 'es', 's'):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [simple_stem(w) for w in tokens if w not in STOP_WORDS and len(w) > 2]
    return ' '.join(tokens)

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_PATH = 'model.pkl'
STATS_PATH = 'model_stats.json'
DATASET_DIR = 'archive'

model_stats = {}

def load_csv_dataset():
    import csv
    X, y = [], []
    for filename, label in [('Fake.csv', 0), ('True.csv', 1)]:
        filepath = os.path.join(DATASET_DIR, filename)
        with open(filepath, encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            for row in reader:
                title = row.get('title', '')
                text = row.get('text', '')
                combined = f"{title} {text}".strip()
                if combined:
                    X.append(combined)
                    y.append(label)
    return X, y

def train_model():
    global model_stats
    print("Cargando dataset...")
    X, y = load_csv_dataset()
    print(f"Dataset cargado: {len(X)} muestras ({y.count(0)} fake, {y.count(1)} real)")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(preprocessor=preprocess_text, max_features=10000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(max_iter=1000, C=1.0))
    ])
    print("Entrenando modelo...")
    pipeline.fit(X_train, y_train)

    print("Evaluando modelo...")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)

    feature_names = pipeline.named_steps['tfidf'].get_feature_names_out()
    coefs = pipeline.named_steps['clf'].coef_[0]

    model_stats = {
        'accuracy': round(accuracy_score(y_test, y_pred) * 100, 1),
        'f1_fake': round(f1_score(y_test, y_pred, pos_label=0) * 100, 1),
        'f1_real': round(f1_score(y_test, y_pred, pos_label=1) * 100, 1),
        'auc_roc': round(roc_auc_score(y_test, y_proba) * 100, 1),
        'confusion_matrix': cm.tolist(),
        'total_samples': len(X),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'fake_count': y.count(0),
        'real_count': y.count(1),
        'top_fake_words': [[feature_names[i], round(float(coefs[i]), 3)] for i in coefs.argsort()[:10]],
        'top_real_words': [[feature_names[i], round(float(coefs[i]), 3)] for i in coefs.argsort()[-10:][::-1]],
        'trained_at': time.strftime('%Y-%m-%d %H:%M'),
    }

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(pipeline, f)
    with open(STATS_PATH, 'w', encoding='utf-8') as f:
        json.dump(model_stats, f)
    print("Modelo guardado en model.pkl")
    return pipeline

def load_model():
    global model_stats
    if os.path.exists(STATS_PATH):
        with open(STATS_PATH, 'r', encoding='utf-8') as f:
            model_stats = json.load(f)
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return train_model()

model = load_model()

# ── Analysis helpers ─────────────────────────────────────────────────────────
FAKE_SIGNALS = [
    ('CAPS LOCK ABUSE', r'\b[A-Z]{4,}\b', 'Excessive capitalization is a common sensationalism pattern'),
    ('EXCLAMATION OVERUSE', r'!{2,}|(?:!.*){3,}', 'Multiple exclamation marks suggest emotional manipulation'),
    ('URGENCY LANGUAGE', r'\b(breaking|urgent|share now|before it.s deleted|act now|limited time)\b', 'Urgency tactics pressure readers without reflection'),
    ('CONSPIRACY PHRASES', r'\b(they don.t want|hidden truth|wake up|deep state|mainstream media lies|suppressed)\b', 'Classic conspiracy framing language'),
    ('ANONYMOUS SOURCING', r'\b(sources say|experts claim|many people|everyone knows|they say)\b', 'Vague, unverifiable sources undermine credibility'),
    ('EMOTIONAL TRIGGERS', r'\b(shocking|bombshell|outrage|scandal|exposed|treasonous|destroy|evil)\b', 'Heavy emotional language bypasses rational thinking'),
]

REAL_SIGNALS = [
    ('NAMED SOURCES', r'\b(reuters|ap|associated press|according to|said in a statement|told reporters)\b', 'Named, credible sources increase reliability'),
    ('INSTITUTIONAL REF', r'\b(department|university|institute|committee|senate|congress|federal|official)\b', 'References to established institutions suggest accountability'),
    ('HEDGED LANGUAGE', r'\b(may|might|could|according to|suggests|indicates|appears|reportedly)\b', 'Hedged language reflects journalistic caution'),
    ('DATA CITATION', r'\b(percent|study|research|data|statistics|report|survey|findings)\b', 'References to measurable evidence support credibility'),
    ('DATE & CONTEXT', r'\b(monday|tuesday|wednesday|thursday|friday|january|february|march|april|may|june|july|august|september|october|november|december)\b', 'Time-specific references ground stories in verifiable events'),
]

def analyze_signals(text):
    text_lower = text.lower()
    fake_found = []
    real_found = []
    for name, pattern, desc in FAKE_SIGNALS:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        if matches:
            fake_found.append({'name': name, 'count': len(matches), 'description': desc})
    for name, pattern, desc in REAL_SIGNALS:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        if matches:
            real_found.append({'name': name, 'count': len(matches), 'description': desc})
    return fake_found, real_found

def get_key_words(text, n=8):
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    words = [w for w in words if w not in STOP_WORDS]
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    return sorted(freq.items(), key=lambda x: -x[1])[:n]

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text', '').strip()

    if not text or len(text) < 20:
        return jsonify({'error': 'Please enter at least 20 characters of news text.'}), 400

    # Simulate slight processing delay for UX
    time.sleep(0.4)

    # Model prediction
    proba = model.predict_proba([text])[0]
    fake_prob = float(proba[0])
    real_prob = float(proba[1])
    label = 'REAL' if real_prob > fake_prob else 'FAKE'
    confidence = max(fake_prob, real_prob) * 100

    # Signal analysis
    fake_signals, real_signals = analyze_signals(text)
    key_words = get_key_words(text)

    # Risk level
    if confidence >= 90:
        level = 'very_high'
        level_text = 'Very High Confidence'
    elif confidence >= 75:
        level = 'high'
        level_text = 'High Confidence'
    elif confidence >= 60:
        level = 'moderate'
        level_text = 'Moderate Confidence'
    else:
        level = 'low'
        level_text = 'Low Confidence — Borderline Case'

    word_count = len(text.split())

    return jsonify({
        'label': label,
        'confidence': round(confidence, 1),
        'fake_prob': round(fake_prob * 100, 1),
        'real_prob': round(real_prob * 100, 1),
        'level': level,
        'level_text': level_text,
        'fake_signals': fake_signals,
        'real_signals': real_signals,
        'key_words': key_words,
        'word_count': word_count,
        'char_count': len(text),
    })

@app.route('/model-info')
def model_info():
    return jsonify(model_stats)

@app.route('/retrain', methods=['POST'])
def retrain():
    global model
    model = train_model()
    return jsonify({'status': 'ok', **model_stats})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
