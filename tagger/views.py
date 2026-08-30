from django.shortcuts import render
from collections import Counter, defaultdict
from pathlib import Path

BASE       = Path(__file__).resolve().parent
CORPUS_DIR = BASE / 'corpus'

START, END = '<START>', '<END>'

def ler_lexico_fwdata(caminho):
    lexico = defaultdict(Counter)
    try:
        from lxml import etree
        tree  = etree.parse(caminho)
        root  = tree.getroot()
        index = {rt.get('guid'): rt for rt in root.findall('rt')}

        def get_pos(guid):
            obj = index.get(guid)
            if obj is None: return None
            abbr = obj.find('.//Abbreviation/AUni[@ws="en"]')
            if abbr is not None and abbr.text: return abbr.text.strip()
            name = obj.find('.//Name/AUni[@ws="en"]')
            if name is not None and name.text: return name.text.strip()
            return None

        def get_form(guid):
            obj = index.get(guid)
            if obj is None: return None
            for form in obj.findall('.//Form/AUni'):
                if form.text: return form.text.strip()
            return None

        for rt in root.findall('rt'):
            if rt.get('class') != 'LexEntry': continue
            lf = rt.find('.//LexemeForm/objsur')
            if lf is None: continue
            forma = get_form(lf.get('guid'))
            if not forma: continue
            pos = None
            for ref in rt.findall('.//MorphoSyntaxAnalyses/objsur'):
                msa = index.get(ref.get('guid'))
                if msa is None: continue
                pr = msa.find('.//PartOfSpeech/objsur')
                if pr is not None:
                    pos = get_pos(pr.get('guid'))
                    if pos: break
            if not pos:
                for ref in rt.findall('.//Senses/objsur'):
                    sense = index.get(ref.get('guid'))
                    if sense is None: continue
                    mref = sense.find('.//MorphoSyntaxAnalysis/objsur')
                    if mref is not None:
                        msa = index.get(mref.get('guid'))
                        if msa is not None:
                            pr = msa.find('.//PartOfSpeech/objsur')
                            if pr is not None:
                                pos = get_pos(pr.get('guid'))
                                if pos: break
            if pos:
                lexico[forma.lower()][pos] += 1
    except Exception as e:
        print(f"Erro fwdata: {e}")
    return lexico

def ler_flextext_treino(caminho):
    frases, lexico = [], defaultdict(Counter)
    try:
        from lxml import etree
        tree = etree.parse(caminho)
        root = tree.getroot()
        for phrase in root.iter('phrase'):
            atual = []
            for word in phrase.findall('words/word'):
                txt = word.find('item[@type="txt"]')
                pos = word.find('item[@type="pos"]')
                if txt is None or not txt.text: continue
                token = txt.text.strip()
                tag   = pos.text.strip() if pos is not None and pos.text else None
                if tag:
                    atual.append((token, tag))
                    lexico[token.lower()][tag] += 1
            if atual:
                frases.append(atual)
    except Exception as e:
        print(f"Erro flextext treino: {e}")
    return frases, lexico

def ler_arquivo_usuario(arquivo):
    nome     = arquivo.name.lower()
    conteudo = arquivo.read().decode('utf-8', errors='ignore')
    tokens   = []
    tem_gabarito = False

    if nome.endswith('.conllu'):
        for linha in conteudo.splitlines():
            linha = linha.strip()
            if not linha or linha.startswith('#'): continue
            cols = linha.split('\t')
            if len(cols) < 4: continue
            if '-' in cols[0] or '.' in cols[0]: continue
            if cols[3] and cols[3] != '_':
                tokens.append((cols[1], cols[3]))
                tem_gabarito = True
            else:
                tokens.append((cols[1], None))

    elif nome.endswith('.flextext'):
        try:
            from lxml import etree
            import io
            tree = etree.parse(io.StringIO(conteudo))
            root = tree.getroot()
            for word in root.iter('word'):
                txt = word.find('item[@type="txt"]')
                pos = word.find('item[@type="pos"]')
                if txt is None or not txt.text: continue
                tag = pos.text.strip() if pos is not None and pos.text else None
                tokens.append((txt.text.strip(), tag))
                if tag: tem_gabarito = True
        except Exception as e:
            print(f"Erro flextext enviado: {e}")
    else:
        for linha in conteudo.splitlines():
            for palavra in linha.split():
                if palavra.strip():
                    tokens.append((palavra.strip(), None))

    return tokens[:200], tem_gabarito

def treinar_trigrama(frases):
    uni, bi, tri = Counter(), Counter(), Counter()
    lex = defaultdict(Counter)
    for frase in frases:
        tags = [START] + [t for _, t in frase] + [END]
        for t in tags[1:-1]: uni[t] += 1
        for i in range(len(tags)-1): bi[f"{tags[i]}→{tags[i+1]}"] += 1
        for i in range(len(tags)-2): tri[f"{tags[i]}→{tags[i+1]}→{tags[i+2]}"] += 1
        for p, t in frase: lex[p.lower()][t] += 1

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

    tag_mf = uni.most_common(1)[0][0] if uni else 'n'
    return prob_tri, prob_bi, lex, tag_mf

def predizer_tag(prob_tri, prob_bi, lex, tag_mf, tag_ant, tag_pos, palavra):
    ctx = (tag_ant, tag_pos)
    if ctx in prob_tri and prob_tri[ctx]:
        return max(prob_tri[ctx], key=prob_tri[ctx].get)
    if tag_ant in prob_bi and prob_bi[tag_ant]:
        return max(prob_bi[tag_ant], key=prob_bi[tag_ant].get)
    if palavra.lower() in lex:
        return lex[palavra.lower()].most_common(1)[0][0]
    return tag_mf

def empacotar(palavras, tags_pred, gabaritos):
    resultado = []
    for palavra, tag, gabarito in zip(palavras, tags_pred, gabaritos):
        acerto = (tag == gabarito) if gabarito else None
        resultado.append({'palavra': palavra, 'tag': tag,
                          'gabarito': gabarito, 'acerto': acerto})
    return resultado

def rodar_trigrama(frases, tokens_com_gabarito):
    prob_tri, prob_bi, lex, tag_mf = treinar_trigrama(frases)
    palavras  = [p for p, _ in tokens_com_gabarito]
    gabaritos = [g for _, g in tokens_com_gabarito]
    tags_pred = []
    for i, palavra in enumerate(palavras):
        tag_ant = tags_pred[i-1] if i > 0 else START
        tag = predizer_tag(prob_tri, prob_bi, lex, tag_mf, tag_ant, END, palavra)
        tags_pred.append(tag)
    return empacotar(palavras, tags_pred, gabaritos)

def rodar_bpe(frases, lexico_ext, tokens_com_gabarito):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    import tempfile, os

    lex = defaultdict(Counter)
    corpus_palavras = []
    for frase in frases:
        for palavra, tag in frase:
            lex[palavra.lower()][tag] += 1
            corpus_palavras.append(palavra)
    for palavra, contagens in lexico_ext.items():
        for tag, count in contagens.items():
            lex[palavra][tag] += count

    tag_mf = (Counter(t for f in frases for _, t in f).most_common(1)[0][0]
              if frases else 'n')

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                     delete=False, encoding='utf-8') as f:
        for p in corpus_palavras: f.write(p + '\n')
        tmp = f.name

    tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=300, min_frequency=2,
                         special_tokens=['[UNK]'])
    tokenizer.train([tmp], trainer)
    os.unlink(tmp)

    resultado = []
    for palavra, gabarito in tokens_com_gabarito:
        enc     = tokenizer.encode(palavra)
        subtoks = enc.tokens
        chave   = palavra.lower()
        tag     = lex[chave].most_common(1)[0][0] if chave in lex else tag_mf
        acerto  = (tag == gabarito) if gabarito else None
        resultado.append({'palavra': palavra, 'tag': tag,
                          'subtokens': subtoks, 'gabarito': gabarito,
                          'acerto': acerto})
    return resultado

def rodar_lexico(frases, lexico_ext, tokens_com_gabarito):
    lex = defaultdict(Counter)
    for frase in frases:
        for palavra, tag in frase: lex[palavra.lower()][tag] += 1
    for palavra, contagens in lexico_ext.items():
        for tag, count in contagens.items(): lex[palavra][tag] += count

    tag_mf = (Counter(t for f in frases for _, t in f).most_common(1)[0][0]
              if frases else 'n')

    resultado = []
    for palavra, gabarito in tokens_com_gabarito:
        chave = palavra.lower()
        if chave in lex:
            tag       = lex[chave].most_common(1)[0][0]
            confianca = 'Alta' if lex[chave].most_common(1)[0][1] > 5 else 'Baixa'
        else:
            tag       = tag_mf
            confianca = 'OOV'
        acerto = (tag == gabarito) if gabarito else None
        resultado.append({'palavra': palavra, 'tag': tag,
                          'confianca': confianca, 'gabarito': gabarito,
                          'acerto': acerto})
    return resultado

def calcular_stats(resultado):
    com_gabarito = [r for r in resultado if r.get('gabarito')]
    if not com_gabarito: return None
    acertos = sum(1 for r in com_gabarito if r.get('acerto'))
    total   = len(com_gabarito)
    return {'acertos': acertos, 'erros': total - acertos,
            'total': total, 'acuracia': round(100 * acertos / total, 1)}

LINGUAS = {
    'tupinamba': {
        'nome': 'Tupinambá', 'familia': 'Tupí-Guaraní',
        'arquivo': 'Tupinambá.flextext', 'tipo': 'flextext',
        'modelos': ['trigrama', 'bpe', 'lexico'],
    },
    'kikongo': {
        'nome': 'Kikongo', 'familia': 'Bantu (Zona H)',
        'arquivo': 'Kikongo.fwdata', 'tipo': 'fwdata',
        'modelos': ['lexico'],
    },
    'kimbundu': {
        'nome': 'Kimbundu', 'familia': 'Bantu (Zona H)',
        'arquivo': 'Kimbundu.fwdata', 'tipo': 'fwdata',
        'modelos': ['lexico'],
    },
    'dzubukua': {
        'nome': 'Dzubukua', 'familia': 'Macro-Jê / Karirí',
        'arquivo': 'Dzubukua.fwdata', 'tipo': 'fwdata',
        'modelos': ['lexico'],
    },
    'kipea': {
        'nome': 'Kipea', 'familia': 'Macro-Jê / Karirí',
        'arquivo': 'Kipea.fwdata', 'tipo': 'fwdata',
        'modelos': ['lexico'],
    },
}

_cache = {}

def carregar_lingua(lingua_key):
    if lingua_key in _cache: return _cache[lingua_key]
    lingua  = LINGUAS[lingua_key]
    caminho = CORPUS_DIR / lingua['arquivo']
    if lingua['tipo'] == 'flextext':
        frases, lexico = ler_flextext_treino(caminho)
    else:
        frases = []
        lexico = ler_lexico_fwdata(caminho)
    _cache[lingua_key] = (frases, lexico)
    return frases, lexico

def home(request):
    return render(request, 'tagger/home.html', {'linguas': LINGUAS})

def analisar(request):
    if request.method != 'POST':
        return render(request, 'tagger/home.html', {'linguas': LINGUAS})

    lingua_key = request.POST.get('lingua', 'tupinamba')
    modelo_key = request.POST.get('modelo', 'lexico')
    texto      = request.POST.get('texto', '').strip()
    arquivo    = request.FILES.get('arquivo')

    lingua = LINGUAS.get(lingua_key, LINGUAS['tupinamba'])
    frases, lexico = carregar_lingua(lingua_key)

    tokens_com_gabarito = []
    tem_gabarito        = False
    nome_arquivo        = None

    if arquivo:
        nome_arquivo = arquivo.name
        tokens_com_gabarito, tem_gabarito = ler_arquivo_usuario(arquivo)
    elif texto:
        tokens_com_gabarito = [(p, None) for p in texto.split()]
    else:
        return render(request, 'tagger/home.html', {
            'linguas': LINGUAS, 'erro': 'Envie um arquivo ou digite um texto.'})

    if not tokens_com_gabarito:
        return render(request, 'tagger/home.html', {
            'linguas': LINGUAS, 'erro': 'Nenhum token encontrado.'})

    resultado_tri = resultado_bpe = resultado_lex = aviso = stats = None

    if modelo_key == 'trigrama':
        if 'trigrama' in lingua['modelos'] and frases:
            resultado_tri = rodar_trigrama(frases, tokens_com_gabarito)
            stats = calcular_stats(resultado_tri)
        else:
            aviso = f"Trigramas não disponível para {lingua['nome']}. Usando léxico."
            resultado_lex = rodar_lexico(frases, lexico, tokens_com_gabarito)
            stats = calcular_stats(resultado_lex)
            modelo_key = 'lexico'

    elif modelo_key == 'bpe':
        if 'bpe' in lingua['modelos'] and frases:
            resultado_bpe = rodar_bpe(frases, lexico, tokens_com_gabarito)
            stats = calcular_stats(resultado_bpe)
        else:
            aviso = f"BPE não disponível para {lingua['nome']}. Usando léxico."
            resultado_lex = rodar_lexico(frases, lexico, tokens_com_gabarito)
            stats = calcular_stats(resultado_lex)
            modelo_key = 'lexico'

    elif modelo_key == 'lexico':
        resultado_lex = rodar_lexico(frases, lexico, tokens_com_gabarito)
        stats = calcular_stats(resultado_lex)

    return render(request, 'tagger/resultado.html', {
        'lingua': lingua, 'lingua_key': lingua_key, 'modelo_key': modelo_key,
        'texto': texto, 'nome_arquivo': nome_arquivo, 'tem_gabarito': tem_gabarito,
        'stats': stats, 'resultado_tri': resultado_tri,
        'resultado_bpe': resultado_bpe, 'resultado_lex': resultado_lex,
        'aviso': aviso, 'linguas': LINGUAS,
    })
