#!/bin/bash
# Script para configurar o projeto LinguaTagger
# Roda de dentro da pasta ~/LinguaTagger com o venv ativado

# ── settings.py ──────────────────────────────────────────────────
cat > LinguaTagger/settings.py << 'EOF'
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SECRET_KEY = 'lingua-tagger-secret-key-2025'
DEBUG = True
ALLOWED_HOSTS = ['*']

INSTALLED_APPS = [
    'django.contrib.staticfiles',
    'tagger',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.middleware.common.CommonMiddleware',
]

ROOT_URLCONF = 'LinguaTagger.urls'

TEMPLATES = [{
    'BACKEND': 'django.template.backends.django.DjangoTemplates',
    'DIRS': [],
    'APP_DIRS': True,
    'OPTIONS': {
        'context_processors': [
            'django.template.context_processors.request',
        ],
    },
}]

STATIC_URL = '/static/'
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
EOF

# ── urls principal ────────────────────────────────────────────────
cat > LinguaTagger/urls.py << 'EOF'
from django.urls import path, include

urlpatterns = [
    path('', include('tagger.urls')),
]
EOF

# ── urls do app ───────────────────────────────────────────────────
cat > tagger/urls.py << 'EOF'
from django.urls import path
from . import views

urlpatterns = [
    path('',        views.home,      name='home'),
    path('analisar/', views.analisar, name='analisar'),
]
EOF

# ── views.py ─────────────────────────────────────────────────────
cat > tagger/views.py << 'EOF'
from django.shortcuts import render
from collections import Counter, defaultdict
from pathlib import Path
import json, os

BASE = Path(__file__).resolve().parent

# ----------------------------------------------------------------
# Utilitário: ler CoNLL-U
# ----------------------------------------------------------------
def ler_conllu(caminho):
    frases, atual = [], []
    try:
        with open(caminho, encoding='utf-8') as f:
            for linha in f:
                linha = linha.rstrip('\n')
                if not linha.strip():
                    if atual:
                        frases.append(atual)
                    atual = []
                    continue
                if linha.startswith('#'):
                    continue
                cols = linha.split('\t')
                if len(cols) < 4:
                    continue
                if '-' in cols[0] or '.' in cols[0]:
                    continue
                if cols[3] and cols[3] != '_':
                    atual.append((cols[1], cols[3]))
        if atual:
            frases.append(atual)
    except FileNotFoundError:
        pass
    return frases

# ----------------------------------------------------------------
# Utilitário: ler FLEx (.flextext) para léxico
# ----------------------------------------------------------------
def ler_lexico_flex(caminho):
    lexico = defaultdict(Counter)
    try:
        from lxml import etree
        tree = etree.parse(caminho)
        root = tree.getroot()
        for word in root.iter('word'):
            txt = word.find('item[@type="txt"]')
            pos = word.find('item[@type="pos"]')
            if txt is not None and txt.text and pos is not None and pos.text:
                lexico[txt.text.strip().lower()][pos.text.strip()] += 1
    except Exception:
        pass
    return lexico

# ----------------------------------------------------------------
# Treinar modelo de trigramas
# ----------------------------------------------------------------
START, END = '<START>', '<END>'

def treinar_trigrama(frases):
    uni, bi, tri = Counter(), Counter(), Counter()
    lexico = defaultdict(Counter)
    for frase in frases:
        tags = [START] + [t for _, t in frase] + [END]
        for t in tags[1:-1]:
            uni[t] += 1
        for i in range(len(tags)-1):
            bi[f"{tags[i]}→{tags[i+1]}"] += 1
        for i in range(len(tags)-2):
            tri[f"{tags[i]}→{tags[i+1]}→{tags[i+2]}"] += 1
        for palavra, tag in frase:
            lexico[palavra.lower()][tag] += 1

    prob_tri = defaultdict(lambda: defaultdict(float))
    for s, c in tri.items():
        if c < 2: continue
        p = s.split('→')
        if len(p) != 3 or p[1] in (START, END): continue
        prob_tri[(p[0], p[2])][p[1]] += c
    for ctx in prob_tri:
        t = sum(prob_tri[ctx].values())
        for tag in prob_tri[ctx]: prob_tri[ctx][tag] /= t

    prob_bi = defaultdict(lambda: defaultdict(float))
    for s, c in bi.items():
        if c < 2: continue
        p = s.split('→')
        if len(p) != 2 or p[1] in (START, END): continue
        prob_bi[p[0]][p[1]] += c
    for ctx in prob_bi:
        t = sum(prob_bi[ctx].values())
        for tag in prob_bi[ctx]: prob_bi[ctx][tag] /= t

    tag_mf = uni.most_common(1)[0][0] if uni else 'NOUN'
    return prob_tri, prob_bi, lexico, tag_mf

def predizer_oraculo(prob_tri, prob_bi, lexico, tag_mf, tag_ant, tag_pos, palavra):
    ctx = (tag_ant, tag_pos)
    if ctx in prob_tri and prob_tri[ctx]:
        return max(prob_tri[ctx], key=prob_tri[ctx].get), 'Trigrama'
    if tag_ant in prob_bi and prob_bi[tag_ant]:
        return max(prob_bi[tag_ant], key=prob_bi[tag_ant].get), 'Bigrama'
    chave = palavra.lower()
    if chave in lexico:
        return lexico[chave].most_common(1)[0][0], 'Léxico'
    return tag_mf, 'Unigrama'

def tagger_trigrama(frases_treino, tokens):
    prob_tri, prob_bi, lexico, tag_mf = treinar_trigrama(frases_treino)
    tags_pred = []
    n = len(tokens)
    for i, token in enumerate(tokens):
        tag_ant = tags_pred[i-1] if i > 0 else START
        tag_pos = END
        # para oráculo: tenta usar tag real posterior se disponível
        tag, metodo = predizer_oraculo(prob_tri, prob_bi, lexico, tag_mf,
                                        tag_ant, tag_pos, token)
        tags_pred.append(tag)
    return list(zip(tokens, tags_pred))

# ----------------------------------------------------------------
# Modelo BPE simples
# ----------------------------------------------------------------
def tagger_bpe(frases_treino, tokens):
    """
    Segmenta cada token com BPE treinado no corpus,
    depois atribui a tag mais comum do léxico para cada token.
    """
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    import tempfile, os

    lexico = defaultdict(Counter)
    corpus_palavras = []
    for frase in frases_treino:
        for palavra, tag in frase:
            lexico[palavra.lower()][tag] += 1
            corpus_palavras.append(palavra)

    tag_mf = Counter(t for f in frases_treino for _, t in f).most_common(1)[0][0]

    # Salva corpus temporário para treinar BPE
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                     delete=False, encoding='utf-8') as f:
        for p in corpus_palavras:
            f.write(p + '\n')
        tmp = f.name

    tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=500, min_frequency=2,
                         special_tokens=['[UNK]'])
    tokenizer.train([tmp], trainer)
    os.unlink(tmp)

    resultado = []
    for token in tokens:
        enc = tokenizer.encode(token)
        subtokens = enc.tokens
        chave = token.lower()
        tag = lexico[chave].most_common(1)[0][0] if chave in lexico else tag_mf
        resultado.append((token, tag, subtokens))

    return resultado

# ----------------------------------------------------------------
# Modelo Léxico (baseline)
# ----------------------------------------------------------------
def tagger_lexico(frases_treino, lexico_flex, tokens):
    lexico = defaultdict(Counter)
    for frase in frases_treino:
        for palavra, tag in frase:
            lexico[palavra.lower()][tag] += 1
    # Incorpora léxico do FLEx
    for palavra, contagens in lexico_flex.items():
        for tag, count in contagens.items():
            lexico[palavra][tag] += count

    tag_mf = Counter(t for f in frases_treino for _, t in f).most_common(1)[0][0] if frases_treino else 'NOUN'

    resultado = []
    for token in tokens:
        chave = token.lower()
        if chave in lexico:
            tag = lexico[chave].most_common(1)[0][0]
            confianca = 'Alta' if lexico[chave].most_common(1)[0][1] > 5 else 'Baixa'
        else:
            tag = tag_mf
            confianca = 'OOV'
        resultado.append((token, tag, confianca))
    return resultado

# ----------------------------------------------------------------
# Configuração das línguas
# ----------------------------------------------------------------
CORPUS_DIR = BASE / 'corpus'

LINGUAS = {
    'tupinamba': {
        'nome': 'Tupinambá',
        'familia': 'Tupí-Guaraní',
        'conllu': None,
        'flextext': 'Tupinambá.flextext',
        'modelos': ['trigrama', 'bpe', 'lexico'],
    },
    'kikongo': {
        'nome': 'Kikongo',
        'familia': 'Bantu (Zona H)',
        'conllu': None,
        'fwdata': 'Kikongo.fwdata',
        'modelos': ['lexico'],
    },
    'kimbundu': {
        'nome': 'Kimbundu',
        'familia': 'Bantu (Zona H)',
        'conllu': None,
        'fwdata': 'Kimbundu_Project-01.fwdata',
        'modelos': ['lexico'],
    },
    'dzubukua': {
        'nome': 'Dzubukua',
        'familia': 'Macro-Jê / Karirí',
        'conllu': None,
        'fwdata': 'Dzubukua.fwdata',
        'modelos': ['lexico'],
    },
    'kipea': {
        'nome': 'Kipea',
        'familia': 'Macro-Jê / Karirí',
        'conllu': None,
        'fwdata': 'KIPEA_.fwdata',
        'modelos': ['lexico'],
    },
    'portugues': {
        'nome': 'Português',
        'familia': 'Indo-Europeia / Romance',
        'conllu': 'pt_bosque-ud-train.conllu',
        'modelos': ['trigrama', 'bpe', 'lexico'],
    },
}

# ----------------------------------------------------------------
# Views
# ----------------------------------------------------------------
def home(request):
    return render(request, 'tagger/home.html', {'linguas': LINGUAS})

def analisar(request):
    if request.method != 'POST':
        return render(request, 'tagger/home.html', {'linguas': LINGUAS})

    lingua_key = request.POST.get('lingua', 'portugues')
    modelo_key = request.POST.get('modelo', 'trigrama')
    texto      = request.POST.get('texto', '').strip()

    if not texto:
        return render(request, 'tagger/home.html', {
            'linguas': LINGUAS,
            'erro': 'Digite um texto para analisar.'
        })

    tokens = texto.split()
    lingua = LINGUAS.get(lingua_key, LINGUAS['portugues'])

    # Carrega corpus CoNLL-U se existir
    frases_treino = []
    if lingua.get('conllu'):
        caminho = CORPUS_DIR / lingua['conllu']
        frases_treino = ler_conllu(caminho)

    # Carrega léxico FLEx se existir
    lexico_flex = defaultdict(Counter)
    if lingua.get('flextext'):
        lexico_flex = ler_lexico_flex(CORPUS_DIR / lingua['flextext'])

    # Roda o modelo selecionado
    resultado_tri  = None
    resultado_bpe  = None
    resultado_lex  = None
    aviso          = None

    if modelo_key == 'trigrama':
        if frases_treino:
            resultado_tri = tagger_trigrama(frases_treino, tokens)
        else:
            aviso = f"O modelo de Trigramas não está disponível para {lingua['nome']} — sem corpus de frases anotado."
            resultado_lex = tagger_lexico(frases_treino, lexico_flex, tokens)
            modelo_key = 'lexico'

    elif modelo_key == 'bpe':
        if frases_treino:
            resultado_bpe = tagger_bpe(frases_treino, tokens)
        else:
            aviso = f"O modelo BPE não está disponível para {lingua['nome']} — sem corpus de frases anotado."
            resultado_lex = tagger_lexico(frases_treino, lexico_flex, tokens)
            modelo_key = 'lexico'

    elif modelo_key == 'lexico':
        resultado_lex = tagger_lexico(frases_treino, lexico_flex, tokens)

    return render(request, 'tagger/resultado.html', {
        'lingua':       lingua,
        'lingua_key':   lingua_key,
        'modelo_key':   modelo_key,
        'texto':        texto,
        'tokens':       tokens,
        'resultado_tri': resultado_tri,
        'resultado_bpe': resultado_bpe,
        'resultado_lex': resultado_lex,
        'aviso':        aviso,
        'linguas':      LINGUAS,
    })
EOF

# ── Criar pasta de templates e corpus ────────────────────────────
mkdir -p tagger/templates/tagger
mkdir -p tagger/corpus

# ── base.html ────────────────────────────────────────────────────
cat > tagger/templates/tagger/base.html << 'EOF'
<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>LinguaTagger</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:       #0f1117;
      --surface:  #1a1d27;
      --border:   #2a2d3a;
      --accent:   #4f8ef7;
      --accent2:  #7c5cbf;
      --text:     #e2e6f0;
      --muted:    #7a7f94;
      --success:  #3ecf8e;
      --warning:  #f5a623;
      --radius:   10px;
      --font:     'Segoe UI', system-ui, sans-serif;
    }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: var(--font);
      min-height: 100vh;
    }

    nav {
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      padding: 0 2rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      height: 56px;
    }

    .nav-brand {
      font-weight: 700;
      font-size: 1.1rem;
      color: var(--accent);
      text-decoration: none;
      letter-spacing: 0.02em;
    }

    .nav-sub {
      font-size: 0.78rem;
      color: var(--muted);
    }

    main {
      max-width: 900px;
      margin: 0 auto;
      padding: 2.5rem 1.5rem;
    }

    h1 {
      font-size: 1.8rem;
      font-weight: 700;
      margin-bottom: 0.4rem;
    }

    h2 {
      font-size: 1.1rem;
      font-weight: 600;
      margin-bottom: 1rem;
      color: var(--text);
    }

    .subtitle {
      color: var(--muted);
      font-size: 0.95rem;
      margin-bottom: 2rem;
    }

    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 1.5rem;
      margin-bottom: 1.5rem;
    }

    label {
      display: block;
      font-size: 0.85rem;
      color: var(--muted);
      margin-bottom: 0.4rem;
      font-weight: 500;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    select, textarea {
      width: 100%;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 6px;
      color: var(--text);
      padding: 0.6rem 0.8rem;
      font-size: 0.95rem;
      font-family: var(--font);
      outline: none;
      transition: border-color 0.2s;
      margin-bottom: 1.2rem;
    }

    select:focus, textarea:focus {
      border-color: var(--accent);
    }

    textarea { resize: vertical; min-height: 90px; }

    .btn {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.6rem 1.4rem;
      border-radius: 6px;
      border: none;
      font-size: 0.95rem;
      font-weight: 600;
      cursor: pointer;
      transition: opacity 0.2s;
      text-decoration: none;
    }

    .btn:hover { opacity: 0.85; }
    .btn-primary { background: var(--accent); color: #fff; }
    .btn-ghost {
      background: transparent;
      border: 1px solid var(--border);
      color: var(--muted);
    }

    .badge {
      display: inline-block;
      padding: 0.15rem 0.55rem;
      border-radius: 20px;
      font-size: 0.75rem;
      font-weight: 600;
    }

    .badge-noun  { background: #1e3a5f; color: #7ab3f0; }
    .badge-verb  { background: #1e3d2f; color: #5ecf9a; }
    .badge-adj   { background: #3d2a1e; color: #f0a96a; }
    .badge-adv   { background: #2d1e3d; color: #b07af0; }
    .badge-pron  { background: #1e3535; color: #6ae0e0; }
    .badge-det   { background: #3d3520; color: #e0c96a; }
    .badge-adp   { background: #3d2030; color: #f07aaa; }
    .badge-other { background: #252830; color: var(--muted); }

    .alert {
      padding: 0.8rem 1rem;
      border-radius: 6px;
      font-size: 0.9rem;
      margin-bottom: 1.2rem;
    }

    .alert-warning {
      background: #2d2210;
      border: 1px solid #6b4c10;
      color: var(--warning);
    }

    .token-grid {
      display: flex;
      flex-wrap: wrap;
      gap: 0.6rem;
      margin-top: 0.5rem;
    }

    .token-item {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 0.3rem;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.6rem 0.8rem;
      min-width: 70px;
    }

    .token-word {
      font-size: 0.95rem;
      font-weight: 600;
    }

    .sub-list {
      display: flex;
      flex-wrap: wrap;
      gap: 0.25rem;
      justify-content: center;
    }

    .sub-tok {
      font-size: 0.7rem;
      background: #252830;
      border: 1px solid var(--border);
      border-radius: 4px;
      padding: 0.1rem 0.3rem;
      color: var(--muted);
      font-family: monospace;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
    }

    th {
      text-align: left;
      padding: 0.6rem 0.8rem;
      color: var(--muted);
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      border-bottom: 1px solid var(--border);
    }

    td {
      padding: 0.6rem 0.8rem;
      border-bottom: 1px solid var(--border);
    }

    tr:last-child td { border-bottom: none; }

    .tag-desc {
      color: var(--muted);
      font-size: 0.82rem;
    }

    footer {
      text-align: center;
      padding: 2rem;
      color: var(--muted);
      font-size: 0.8rem;
      border-top: 1px solid var(--border);
      margin-top: 3rem;
    }
  </style>
</head>
<body>
<nav>
  <a href="/" class="nav-brand">LinguaTagger</a>
  <span class="nav-sub">UFBA — Iniciação Científica em PLN</span>
</nav>
<main>
  {% block content %}{% endblock %}
</main>
<footer>
  Universidade Federal da Bahia &nbsp;·&nbsp; Instituto de Ciência da Computação &nbsp;·&nbsp; 2025
</footer>
</body>
</html>
EOF

# ── home.html ─────────────────────────────────────────────────────
cat > tagger/templates/tagger/home.html << 'EOF'
{% extends "tagger/base.html" %}
{% block content %}

<h1>LinguaTagger</h1>
<p class="subtitle">
  Ferramenta de anotação morfossintática para línguas de baixo recurso.<br>
  Selecione a língua, o modelo e insira o texto para analisar.
</p>

{% if erro %}
<div class="alert alert-warning">{{ erro }}</div>
{% endif %}

<form method="post" action="/analisar/">
  {% csrf_token %}

  <div class="card">
    <h2>Configuração</h2>

    <label for="lingua">Língua</label>
    <select name="lingua" id="lingua">
      {% for key, info in linguas.items %}
      <option value="{{ key }}">{{ info.nome }} — {{ info.familia }}</option>
      {% endfor %}
    </select>

    <label for="modelo">Modelo</label>
    <select name="modelo" id="modelo">
      <option value="trigrama">Trigramas com Backoff</option>
      <option value="bpe">BPE + Léxico</option>
      <option value="lexico">Léxico (baseline)</option>
    </select>

    <label for="texto">Texto para analisar</label>
    <textarea name="texto" id="texto"
      placeholder="Digite ou cole o texto aqui..."></textarea>

    <button type="submit" class="btn btn-primary">Analisar →</button>
  </div>
</form>

<div class="card">
  <h2>Sobre os modelos</h2>
  <table>
    <thead>
      <tr>
        <th>Modelo</th>
        <th>Como funciona</th>
        <th>Disponível para</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Trigramas</strong></td>
        <td class="tag-desc">Usa o contexto das tags vizinhas para predizer a classe gramatical de cada palavra.</td>
        <td class="tag-desc">Tupinambá, Português</td>
      </tr>
      <tr>
        <td><strong>BPE + Léxico</strong></td>
        <td class="tag-desc">Segmenta cada palavra em subpalavras (morfemas estatísticos) e consulta o léxico.</td>
        <td class="tag-desc">Tupinambá, Português</td>
      </tr>
      <tr>
        <td><strong>Léxico</strong></td>
        <td class="tag-desc">Consulta a tag mais frequente de cada palavra no corpus de treino. Baseline simples.</td>
        <td class="tag-desc">Todas as línguas</td>
      </tr>
    </tbody>
  </table>
</div>

{% endblock %}
EOF

# ── resultado.html ────────────────────────────────────────────────
cat > tagger/templates/tagger/resultado.html << 'EOF'
{% extends "tagger/base.html" %}
{% block content %}

<div style="display:flex; align-items:center; gap:1rem; margin-bottom:1.5rem;">
  <a href="/" class="btn btn-ghost">← Voltar</a>
  <div>
    <h1>{{ lingua.nome }}</h1>
    <p class="subtitle" style="margin:0;">{{ lingua.familia }}</p>
  </div>
</div>

{% if aviso %}
<div class="alert alert-warning">⚠ {{ aviso }}</div>
{% endif %}

<div class="card">
  <h2>Texto analisado</h2>
  <p style="color:var(--muted); font-size:0.9rem;">{{ texto }}</p>
</div>

{% if resultado_tri %}
<div class="card">
  <h2>Resultado — Trigramas com Backoff</h2>
  <p class="tag-desc" style="margin-bottom:1rem;">
    Para cada palavra, o modelo consulta as tags das palavras vizinhas e escolhe a classe gramatical mais provável segundo o corpus de treino.
  </p>
  <div class="token-grid">
    {% for palavra, tag in resultado_tri %}
    <div class="token-item">
      <span class="token-word">{{ palavra }}</span>
      <span class="badge
        {% if tag == 'NOUN' %}badge-noun
        {% elif tag == 'VERB' %}badge-verb
        {% elif tag == 'ADJ' %}badge-adj
        {% elif tag == 'ADV' %}badge-adv
        {% elif tag == 'PRON' %}badge-pron
        {% elif tag == 'DET' %}badge-det
        {% elif tag == 'ADP' %}badge-adp
        {% else %}badge-other{% endif %}
      ">{{ tag }}</span>
    </div>
    {% endfor %}
  </div>
</div>
{% endif %}

{% if resultado_bpe %}
<div class="card">
  <h2>Resultado — BPE + Léxico</h2>
  <p class="tag-desc" style="margin-bottom:1rem;">
    Cada palavra é segmentada em subpalavras pelo algoritmo BPE. A tag é determinada pelo léxico do corpus de treino.
  </p>
  <div class="token-grid">
    {% for palavra, tag, subtokens in resultado_bpe %}
    <div class="token-item">
      <span class="token-word">{{ palavra }}</span>
      <span class="badge
        {% if tag == 'NOUN' %}badge-noun
        {% elif tag == 'VERB' %}badge-verb
        {% elif tag == 'ADJ' %}badge-adj
        {% elif tag == 'ADV' %}badge-adv
        {% elif tag == 'PRON' %}badge-pron
        {% elif tag == 'DET' %}badge-det
        {% elif tag == 'ADP' %}badge-adp
        {% else %}badge-other{% endif %}
      ">{{ tag }}</span>
      <div class="sub-list">
        {% for s in subtokens %}
        <span class="sub-tok">{{ s }}</span>
        {% endfor %}
      </div>
    </div>
    {% endfor %}
  </div>
</div>
{% endif %}

{% if resultado_lex %}
<div class="card">
  <h2>Resultado — Léxico (baseline)</h2>
  <p class="tag-desc" style="margin-bottom:1rem;">
    Cada palavra recebe a tag que teve com maior frequência no corpus de treino. OOV = palavra nunca vista no treino.
  </p>
  <div class="token-grid">
    {% for palavra, tag, confianca in resultado_lex %}
    <div class="token-item">
      <span class="token-word">{{ palavra }}</span>
      <span class="badge
        {% if tag == 'NOUN' %}badge-noun
        {% elif tag == 'VERB' %}badge-verb
        {% elif tag == 'ADJ' %}badge-adj
        {% elif tag == 'ADV' %}badge-adv
        {% elif tag == 'PRON' %}badge-pron
        {% elif tag == 'DET' %}badge-det
        {% elif tag == 'ADP' %}badge-adp
        {% else %}badge-other{% endif %}
      ">{{ tag }}</span>
      <span style="font-size:0.7rem; color:var(--muted);">{{ confianca }}</span>
    </div>
    {% endfor %}
  </div>
</div>
{% endif %}

<div class="card">
  <h2>Legenda das tags</h2>
  <table>
    <thead><tr><th>Tag</th><th>Classe gramatical</th><th>Exemplo</th></tr></thead>
    <tbody>
      <tr><td><span class="badge badge-noun">NOUN</span></td><td>Substantivo</td><td class="tag-desc">governo, casa, tupã</td></tr>
      <tr><td><span class="badge badge-verb">VERB</span></td><td>Verbo</td><td class="tag-desc">correu, sonhei, salva</td></tr>
      <tr><td><span class="badge badge-adj">ADJ</span></td><td>Adjetivo</td><td class="tag-desc">bonito, rápido, catú</td></tr>
      <tr><td><span class="badge badge-adv">ADV</span></td><td>Advérbio</td><td class="tag-desc">ontem, rapidamente</td></tr>
      <tr><td><span class="badge badge-pron">PRON</span></td><td>Pronome</td><td class="tag-desc">eu, ele, nde</td></tr>
      <tr><td><span class="badge badge-det">DET</span></td><td>Determinante</td><td class="tag-desc">o, a, os, as</td></tr>
      <tr><td><span class="badge badge-adp">ADP</span></td><td>Preposição</td><td class="tag-desc">de, em, para</td></tr>
      <tr><td><span class="badge badge-other">PUNCT</span></td><td>Pontuação</td><td class="tag-desc">. , ;</td></tr>
    </tbody>
  </table>
</div>

{% endblock %}
EOF

echo "Todos os arquivos criados com sucesso!"
