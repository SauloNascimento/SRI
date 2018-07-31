import pandas as pd
import math
from nltk.tokenize import RegexpTokenizer
import string

class WordInfo:
    'Classe que guarda informações sobre a palavra, como documentos em que ocorre, tf em cada documento e calculo de idf'

    def __init__(self, word):  # Contrutor da classe recebe apenas a palavra em questão
        self.word = word
        self.idf = 0
        self.docs = {}  # Dicionario que mapeia doc_id ao tf da palavra

    '''Incrementa o TF da palavra em um documento'''

    def found(self, doc_id):  # Metodo que realiza a contagem do tf da palavra
        if (doc_id in self.docs):  # Se o doc_id está mapeado, incrementa-o
            self.docs[doc_id] += 1
        else:
            self.docs[doc_id] = 1  # Caso contrario, define-o como 1

    def calculateIDF(self, totaldocs):  # Metodo de calculo do idf, recebendo o total de documentos
        df = len(self.docs)  # Numero de documentos em que a palavra ocorre
        if (df > 0):
            self.idf = math.log(((totaldocs + 1) / df))

    def getIds(self):  # Metodo que retorna todos os doc_ids em que a palavra ocorre
        return list(self.docs.keys())

    def getTf(self, doc_id):  # Metodo que retorna o tf da palavra em um documento, ou 0 caso não ocorra
        return self.docs.get(doc_id, 0)

# Carregamento dos dados
data = pd.read_csv('app/estadao_noticias_eleicao.csv', encoding="utf-8")
data.fillna('', inplace=True) # Preenchendo os campos vazios da tabela com ''
tokenizer = RegexpTokenizer(r'\w+')
data['tokenized_text'] = data.apply(lambda row: tokenizer.tokenize(row['titulo'] + ' ' + row['subTitulo'] + ' ' + row['conteudo']), axis=1)

# Criação dos conjuntos de teste e validação
train = data.sample(frac=0.8)
validation = data.drop(train.index)

vocab = {} # Vocabulario do conjunto de treinamento
docs = {} # Vetores dos documentos

# Montagem do vocabulário e vetores dos documentos do conjunto de treinamento
for index, row in train.iterrows() :
    id = row['idNoticia']
    text = row['tokenized_text']
    docs[id] = {}
    for word in text:
        if (word not in string.punctuation):
            word_l = word.lower()
            if (word_l not in vocab) :
                vocab[word_l] = WordInfo(word_l)
            vocab[word_l].found(id)
            docs[id][word_l] = 1

# Calculo dos idfs de cada palavra mapeada e atualização dos vetores dos documentos
for word in vocab.keys() :
    vocab[word].calculateIDF(len(docs))
    idf = vocab[word].idf
    for doc_i in docs.keys() :
        if (word in docs[doc_i]) :
            docs[doc_i][word] = vocab[word].getTf(doc_i) * idf

'''Calcula a similaridade entre 2 documentos, representados como um vetor, atravez do produto escalar entre os vetores'''
def similarity(docQ, docI) :
    score = 0
    for word in docQ.keys() :
        score += docQ[word] * docI.get(word, 0)
    return score

'''Transforma o documento em vetor'''
def docVect(index) :
    vect = {}
    text = data[data.idNoticia == index].iloc[0]['tokenized_text']
    for word in text:
        if (word not in string.punctuation):
            word_l = word.lower()
            if (word_l not in vect) :
                vect[word_l] = 0
            if (word_l in vocab) :
                vect[word_l] += vocab[word_l].idf
    return vect

'''Calcula os 5 vizinhos mais próximos de um documento passado através do idNoticia'''
def top5(index) :
    docQ = docVect(index)
    top = [(id_doc, similarity(docQ, docs[id_doc])) for id_doc in list(docs.keys())[:5]]
    top = sorted(top, reverse=True, key=lambda tup: tup[1])
    for i in list(docs.keys())[5:]:
        sim = similarity(docQ, docs[i])
        j = 4
        while (sim > top[j][1] and j >= 0) :
            j -= 1
        top.insert(j+1, (i, sim))
        top = top[:5]
    return [t[0] for t in top[:5]]

'''Retorna os 5 documentos mais semelhantes ao socumento passado e seus dados (titulo, conteudo...)'''
def get_5_neighbors(index) :
    top = top5(index)
    print(top)
    neighbors = []
    for newID in top:
        doc = get_new(newID)
        neighbors.append(doc)
    print(neighbors)
    return neighbors

def get_new(index) :
    doc = {}
    data_doc = data[data.idNoticia == index].iloc[0]
    doc['ID'] = str(data_doc['idNoticia'])
    doc['Titulo'] = str(data_doc['titulo'])
    doc['SubTitulo'] = str(data_doc['subTitulo'])
    doc['Conteudo'] = str(data_doc['conteudo'])[:400] + '...'
    return doc

def getValidationDocs() :
    return list(validation.filter(items=['idNoticia', 'titulo']).to_records(index=False))


def main() :
    get_5_neighbors(index)

if __name__ == "__main__":
    main()
