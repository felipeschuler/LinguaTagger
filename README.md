# LinguaTagger: Ferramenta de Anotação Morfossintática para Línguas de Baixo Recurso

**LinguaTagger** é um protótipo de ferramenta web para anotação morfossintática automática de línguas de baixo recurso — línguas indígenas e africanas com poucos dados computacionais disponíveis. A ferramenta aplica três modelos diferentes de Processamento de Linguagem Natural (PLN) e permite carregar arquivos de corpus diretamente para análise.

Desenvolvido como entregável da Iniciação Científica no **Instituto de Ciência da Computação da Universidade Federal da Bahia (UFBA)**, 2025–2026.

---

## Línguas suportadas

| Língua | Família | Modelos disponíveis |
|--------|---------|-------------------|
| Tupinambá | Tupí-Guaraní | Trigramas, BPE, Léxico |
| Kikongo | Bantu (Zona H) | Trigramas, BPE, Léxico |
| Kimbundu | Bantu (Zona H) | Trigramas, BPE, Léxico |
| Dzubukua | Macro-Jê / Karirí | Trigramas, BPE, Léxico |
| Kipea | Macro-Jê / Karirí | Trigramas, BPE, Léxico |

---

## Modelos implementados

- **Trigramas com Backoff** — usa o contexto das tags vizinhas para predizer a classe gramatical de cada palavra. Backoff em 4 níveis: trigrama → bigrama → léxico → unigrama.
- **BPE + Léxico** — segmenta cada palavra em subpalavras via Byte Pair Encoding (biblioteca `tokenizers` do Hugging Face) e consulta o léxico do corpus para atribuir a tag.
- **Léxico (baseline)** — consulta diretamente a tag mais frequente de cada palavra no corpus de treino. Disponível para todas as línguas.

---

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/felipeschuler/LinguaTagger.git
cd LinguaTagger
```

2. Crie e ative um ambiente virtual:
```bash
python3 -m venv venv_tagger
source venv_tagger/bin/activate
```

3. Instale as dependências:
```bash
pip install django lxml tokenizers morfessor
```

4. Adicione os arquivos de corpus em `tagger/corpus/`:
   - `Tupinambá.flextext` — corpus do Tupinambá em formato FLEx
   - `Kikongo.fwdata`, `Kimbundu.fwdata`, `Dzubukua.fwdata`, `Kipea.fwdata` — léxicos em formato FLEx

   > Os arquivos de corpus não estão incluídos no repositório por serem dados de pesquisa sob responsabilidade da equipe de linguística.

5. Suba o servidor:
```bash
python manage.py runserver
```

6. Acesse no navegador:
```
http://127.0.0.1:8000/
```

---

## Uso

1. Selecione a língua e o modelo na tela inicial
2. Carregue um arquivo (`.flextext`, `.conllu` ou `.txt`) ou digite o texto diretamente
3. Clique em **Analisar**
4. O resultado mostra cada token com sua classe gramatical predita
5. Se o arquivo carregado contiver anotações reais, a ferramenta calcula a acurácia e destaca acertos e erros

---

## Resultados obtidos (Tupinambá — Catecismo Brasílico)

| Modelo | Acurácia |
|--------|----------|
| Trigramas com Backoff | 34,7% |
| BPE + Léxico | 98,3%* |

*O resultado alto do BPE reflete o vocabulário repetitivo do corpus — o léxico conhece a maioria das palavras por terem aparecido no treino. Em textos novos, o desempenho seria menor.

---

## Estrutura do projeto

```
LinguaTagger/
├── manage.py
├── LinguaTagger/
│   ├── settings.py
│   └── urls.py
└── tagger/
    ├── views.py
    ├── urls.py
    ├── corpus/          ← arquivos de dados (não incluídos)
    └── templates/
        └── tagger/
            ├── base.html
            ├── home.html
            └── resultado.html
```

---

## Dependências

- Python 3.14
- Django 6.1
- lxml
- tokenizers (Hugging Face)
- morfessor

---

## Sobre

Iniciação Científica em Processamento de Linguagem Natural para Línguas de Baixo Recurso.

**Orientadores:** Profa. Lilian Teixeira · Prof. Marlo Vieira
**Estudante:** Felipe Schuler Fernandes
**Instituição:** Universidade Federal da Bahia — Instituto de Ciência da Computação
**Período:** 2025–2026
