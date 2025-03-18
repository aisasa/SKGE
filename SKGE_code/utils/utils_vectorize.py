import nltk
import numpy as np
import pickle
from gensim.models.keyedvectors import KeyedVectors

en_stopwords = nltk.corpus.stopwords.words('english')
# Embeddings management
EMBED_DIM = 300
w2v_model = KeyedVectors.load_word2vec_format('embeddings/GoogleNews-vectors-negative300.bin', binary=True)
with open('embeddings/glove_embed_model.pkl', 'rb') as f:
    glove_model = pickle.load(f)
# Unknowns management
try:
    with open('unks_dict.pkl', 'rb') as f:
        unks_dict = pickle.load(f)
except:
    unks_dict = {}


# ====================================Normalization==================================
# sum(v / sum(v)) = 1
def zero_one_normalize(v):
    v = np.array(v)
    v_sum = np.sum(v)
    if v_sum != 0:
        normalized_v = v / v_sum
    else:
        normalized_v = v
    return(np.array(normalized_v))

def sigmoid(a):
    sigmoid = 1.0/(1.0 + np.exp(-a))
    return sigmoid

def gauss(v):
    return(np.exp(-v**2))


# ====================================Similarity=====================================
# Cosine similarity
def cos_sim(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return(0.)
    return(np.dot(a, b)/(np.linalg.norm(a) * np.linalg.norm(b)))

# ICM similarity
def icm_sim(v1, v2, beta=1.2):
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    return((1 - beta)*(v1_norm**2 + v2_norm**2) + beta*(v1@v2))
def picm_sim(v1, v2, beta=1.2):
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    return(max((1 - beta)*(v1_norm**2 + v2_norm**2) + beta*(v1@v2), 0))

# Bidirectional average/median of maximum similirity
def bidir_avgmax_sim(similarity_matrix, stdst='median'):
    matrix = np.array(similarity_matrix)
    n_rows = matrix.shape[0]
    n_cols = matrix.shape[1]
    if stdst == 'mean':
        row_max = np.mean(np.max(matrix, axis=1)) 
        col_max = np.mean(np.max(matrix, axis=0)) 
    if stdst == 'median': 
        row_max = np.median(np.max(matrix, axis=1)) 
        col_max = np.median(np.max(matrix, axis=0)) 
    # Bidirectional similarity average
    return((row_max + col_max) / 2) 

def bertscore(similarity_matrix):
    matrix = np.array(similarity_matrix)
    n_rows = matrix.shape[0]
    n_cols = matrix.shape[1]
    row_max = np.sum(np.max(matrix, axis=1)) / n_rows   
    col_max = np.sum(np.max(matrix, axis=0)) / n_cols  
    if row_max + col_max < 1.e-125: row_max = 1.e-125
    return(2*(row_max * col_max) / (row_max + col_max) )


# ====================================Text treatment=================================
# Receives a word and eliminate '-' or '_' separators
def _normalize_token(txt_token:str):     
    try:
        w2v_model[txt_token]                                    # If token as-is in embed model, ok and return it
        return(txt_token)
    except:                                                     # Token has not embedding
        if '_' not in txt_token and '-' not in txt_token:       # If token does not have separators, return it
            return(txt_token)     
        if '_' in txt_token:                                    # If underscore, change it by a space and return
            txt_token = txt_token.replace('_', ' ')
            return(txt_token)
        if '-' in txt_token:                                    # If hyphen, delete and try if there is embedding
            joint_txt_token = txt_token.replace('-', '')
            try:
                w2v_model[joint_txt_token]
                return(joint_txt_token)
            except:                                             # If not embedding after deletion, change to space and return
                sep_txt_token = txt_token.replace('-', ' ')
                return(sep_txt_token)
# Clean text: treating non-alphabetical chars and apply stop words
def _clean_txt(txt:str, stop_words=False, punct_marks = True):  # Receives text to clean
    sep_chars = '-_'
    # Treating separators '-' and '_'
    txt = ' '.join([_normalize_token(token) for token in txt.split()])
    # Treating punctuation marks
    for char in txt:
        if not (char.isalpha() or char.isnumeric() or char.isspace() or char in sep_chars):
            if punct_marks:
                txt = txt.replace(char, ' '+char+' ')   # Puncts as individual tokens, except sep_chars already treated    
            else:
                txt = txt.replace(char, ' ')            # Delete puncts, except sep_chars 
    # Applying stop words
    if stop_words:
        txt = ' '.join([word for word in txt.split() if word not in en_stopwords]) 

    return(txt)


# =================================Vectors, composition, unknowns================================
# Vectorize (additive or average models)
def vectorize(sent:str, stop_words=False, punct_marks=True, rep_model='add'): # Receives text and representation model
    global w2v_model
    sent = _clean_txt(sent, stop_words=stop_words, punct_marks=punct_marks)
    vector = np.zeros(EMBED_DIM, dtype=float)   # Zeroed base vector with embedding dimension
    tokens = sent.split()
    sent_v = []
    for token in tokens:
        try:
            token_v = w2v_model[token]
        except:
            print('UNK word', token)
            #continue
            token_v = np.full(EMBED_DIM, 0.001, dtype=float)
        if rep_model == 'add':
            vector = vector + token_v           # If additive, add vectors to build final vector
        if rep_model == 'avg': 
            sent_v.append(token_v)              # If average, build a list of word vectors...
            
    if sent_v:                                  # ... and calculate average two by two...
        sent_v_len = len(sent_v)
        pair_avrg = sent_v[0]
        for idx in range(sent_v_len - 1):       # ... left to right...
            pair_avrg = (pair_avrg + sent_v[idx+1]) / 2 
        vector = np.array(pair_avrg)            # ... to build the representative final vector
        
    return(vector)                              # Vector representing input text (a sentence here)


# ICDS generalized composition function; lambda=1 -> F_Inf = ((a+b)/||a+b||)·sqrt(||a||^2 + ||b||^2 - mu·a@b) 
def icds_composition(a, b, mu='ratio'):      
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    #mu = 1.     # F_Joint
    #mu = 0.     # F_Ind
    if mu == 'ratio':
        den_mu = np.max([a_norm, b_norm])
        if den_mu == 0:
            den_mu = 1.e-125    
        mu = np.min([a_norm, b_norm]) / den_mu  # F_Inf
    den_F = np.linalg.norm(a + b)
    if den_F == 0:
        den_F = 1.e-125 
    F = ((a + b) / den_F) * np.sqrt(a_norm**2 + b_norm**2 - mu*(a@b))
    return(F)


def unks(token):
    global unks_dict
    #print('UNK word in embeddings set: ', token)
    if token not in unks_dict:
        '''
        if not token.isalnum():     # Punctuation marks
            v_token = np.random.uniform(-0.0001, 0.0001, 300)
        elif token.isnumeric():     # Numbers
            v_token = np.random.uniform(-1.75, 1.75, 300)     #(-1.75, 1.75, 300)
        elif token not in en_stopwords:
            v_token = np.random.uniform(-2., 2., 300)     # (-2, 2, 300)
        else:
            v_token = np.random.uniform(-0.01, 0.01, 300)   # 0.01
        '''
        v_token = np.random.uniform(-2., 2., 300)
        unks_dict[token] = v_token
        with open('unks_dict.pkl', 'wb') as f:
            pickle.dump(unks_dict, f)
    else:
        #print(token + ' found in unks dictionary')
        v_token = (unks_dict[token])
    return(v_token)


def icds_vectorize(sent:str, stop_words=False, punct_marks=True, embed_model='w2v', mu='ratio'): # Receives text 
    if embed_model == 'w2v': model = w2v_model
    if embed_model == 'glove': model = glove_model
    cleaned_sent = _clean_txt(sent, stop_words=stop_words, punct_marks=punct_marks)
    tokens = cleaned_sent.split()

    if len(tokens) == 0:
        vector = np.zeros((300,))
    elif len(tokens) == 1:
        try:
            vector = model[tokens[0]]
        except:
            vector = unks(tokens[0])
    else:
        for idx in range(len(tokens)-1):   
            if idx == 0:    # First token...
                try:
                    token0_v = model[tokens[idx]]
                except:
                    token0_v = unks(tokens[idx])
                try:        # ... and second token 
                    token1_v = model[tokens[idx+1]]
                except:
                    token1_v = unks(tokens[idx+1])
                vector = icds_composition(token0_v, token1_v, mu)
            else:           # From second token onwards
                try:
                    token = model[tokens[idx+1]]
                except:
                    token = unks(tokens[idx+1])
                vector = icds_composition(vector, token, mu)

    return(vector)                              # Vector representing input text (a sentence here)


def sum_vectorize(sent:str, stop_words=False, punct_marks=True, embed_model='w2v', mu='ratio'): # Receives text 
    #global w2v_model
    if embed_model == 'w2v': model = w2v_model
    if embed_model == 'glove': model = glove_model
    cleaned_sent = _clean_txt(sent, stop_words=stop_words, punct_marks=punct_marks)
    tokens = cleaned_sent.split()
    vector = None

    if len(tokens) == 0:
        #print('VOID result!!! for sent ', sent)
        vector = np.zeros((300,))
    elif len(tokens) == 1:
        try:
            vector = model[tokens[0]]
        except:
            vector = unks(tokens[0])
    else:
        for idx in range(len(tokens)-1):   
            if idx == 0:    # First token...
                try:
                    token0_v = model[tokens[idx]]
                except:
                    token0_v = unks(tokens[idx])
                try:        # ... and second token 
                    token1_v = model[tokens[idx+1]]
                except:
                    token1_v = unks(tokens[idx+1])
                vector = token0_v + token1_v
            else:           # From second token onwards
                try:
                    token = model[tokens[idx+1]]
                except:
                    token = unks(tokens[idx+1])
                vector = vector + token

    return(vector)                              # Vector representing input text (a sentence here)


# ======================================Others=======================================
# Load t-SNE 2D embeddings
def load_2d_tsne(file_name:str):
    keys = w2v_model.index_to_key
    with open(file_name, "rb") as f:            # Example: 'w2v_2d_coord_perp60_niter300.pkl'
        keys, x, y = pickle.load(f)
    embeds_2d = {}
    for idx, key in enumerate(keys):
        embeds_2d[key] = (x[idx], y[idx])
    return(keys, x, y, embeds_2d)