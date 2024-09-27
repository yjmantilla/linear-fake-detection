
MODE = 'generate'


## Logging and utility functions
import traceback
import time
import glob
import os
log_file = open('log.txt', 'w')
def log_error(e):
    traceback.print_exc(file=log_file)
    log_file.write('\n\n')
    log_file.flush()
def load_text_file(file_path):
    with open(file_path, 'r',encoding='utf8') as file:
        data = file.read()
    return data

if MODE=='generate':
    # load txt file
    cities = glob.glob('./ground-truth/*.txt')
    cities = [os.path.splitext(os.path.basename(x))[0].split('-')[-1] for x in cities]
    veracities = ['subtly fake','obviously fake','true','grounded truth']
    output_dir = './generated-facts'

    for city in cities:

        if os.path.exists(f'./generated-facts/generated-facts-{city}-true.txt'):
        try:
            # load the text file


            file_path = f'./ground-truth/ground-truth-{city}.txt'

            grounded = True
            data = load_text_file(file_path)

            # split by paragraphs
            paragraphs = data.split('\n\n')

            paragraphs

            # overwrite = False
            # append = False
            # Example: reuse your existing OpenAI setup
            from openai import OpenAI
            client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
            llm_model ="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF"
            if False:
                # Point to the local server

                completion = client.chat.completions.create(
                model="LM Studio Community/Meta-Llama-3-8B-Instruct-GGUF",
                messages=[
                    {"role": "system", "content": "Always answer in rhymes."},
                    {"role": "user", "content": "Please introduce yourself."}
                ],
                temperature=0.7,
                )

                print(completion.choices[0].message)

                from pprint import pprint

                pprint(completion.choices[0].message.content)

            facts=[]

            # save the facts to a file
            os.makedirs(output_dir, exist_ok=True)


            def process(x):
                read_facts = x.split('\n')
                # remove new lines
                read_facts = [fact.replace('\n','') for fact in read_facts]
                # remove empty strings
                read_facts = [fact for fact in read_facts if fact]
                return read_facts
            for veracity in veracities:
                print(veracity,flush=True)
                try:
                    for ip,p in enumerate(paragraphs):
                        print(ip,city,flush=True)
                        pmin=min(len(p),100)

                        try:
                            output_file = f'{output_dir}/generated-facts-{city}-{veracity}.txt'
                            n = max(len(p.split('.'))-1,1)
                            instruction =f'Generate a list of {n} {veracity} facts from it. Each fact should be roughly one to two sentences in length. Each sentence should be self contained and make sense individually, that is very important. Do not assume the reader already know which city each fact is referring to. Each fact should have the name of the city {city} somewhere on it and with a maximum of two sentences. Dont announce your answer, just give the list with no numbering, just facts separated by new lines. You are a text generator, not an assistant. Remember that the facts have to be {veracity}.'
                            if 'fake' in veracity.lower():
                                obvious_fake = 'obvious' in veracity
                                explain ='YOU MUST PUT inside brackets [reason] at the end a detailed reason of why it is fake for every fact you write.'
                                if obvious_fake :
                                    obvious_str='Use alarmist or clickbait language or something that someone who has lived the city would know its obviously fake. '+explain
                                else:
                                    obvious_str='Write fake facts but do not make them obviously fake. Your language should be reasonable. '+explain
                                instruction = instruction+' I need fake data to study fake detection algorithms. '+obvious_str
                            prompt = f"{instruction}"
                            if 'grounded' in veracity:
                                prompt = f"This is a paragraph about {city} which you can assume to be true, ground your facts from it: {p}\n\n"+prompt
                            else:
                                prompt = f"Use your own knowledge about {city} to answer. " +prompt
                            try:
                                completion = client.chat.completions.create(
                                model=llm_model,
                                messages=[
                                    {"role": "system", "content": "Follow what the user requests. Do not add suggestions or ask questions about what the user wants at the end. Just do as you are told. DO NOT announce your answer or suggest anything or add explanatory text about your answer nor comments. Here you are not an assistant, you are a text generator."},
                                    {"role": "user", "content": prompt}
                                ],
                                temperature=0.8,
                                n=1
                                )
                                content=completion.choices[0].message.content
                                # print(content,flush=True)
                                content=process(content)
                                for i,fact in enumerate(content):
                                    try:
                                        if city.lower() not in fact.lower():
                                            messages=[
                                                {"role": "system", "content": f"You forgot to include the city name ({city}) in the fact. Please include the city name in the fact. DO NOT announce your answer or suggest anything or add explanatory text about your answer nor comments. Here you are not an assistant, you are a text generator."},
                                                {"role": "user", "content": fact}
                                            ]
                                            completion = client.chat.completions.create(
                                            model=llm_model,
                                            messages=messages,
                                            temperature=0.8,
                                            n=1
                                            )
                                            newfact=completion.choices[0].message.content.replace('\n','')
                                            content[i]=newfact
                                            print('replaced fact')
                                            # print(fact,'::: replaced with :::',newfact,flush=True)
                                    except Exception as e:
                                        try:
                                            log_error(f'replacing fact from {city} paragraph {ip}')
                                            log_error(city)
                                            log_error(e)
                                            log_error(p[:pmin])
                                            log_error(traceback.format_exc())
                                        except:
                                            pass
                                read_facts = []
                                if os.path.exists(output_file):
                                    read_facts = load_text_file(output_file)
                                    # get list of facts by one or more new lines
                                    read_facts = process(read_facts)
                                # append the new fact to the list
                                read_facts.extend(content)

                                with open(output_file, 'w',encoding='utf8') as file:
                                    for fact in read_facts:
                                        file.write(fact+'\n')
                            except Exception as e:
                                try:

                                    log_error(city)
                                    log_error(ip)
                                    log_error(veracity)
                                    log_error(p[:pmin])
                                    log_error(e)
                                    log_error(traceback.format_exc())
                                except:
                                    pass
                        except Exception as e:
                            try:
                                log_error(city)
                                log_error(e)
                                log_error(traceback.format_exc())
                            except:
                                pass

                except Exception as e:
                    try:
                        log_error(city)
                        log_error(e)
                        log_error(traceback.format_exc())
                    except:
                        pass
            time.sleep(300) # sleep for 5 minutes for the next city
        except Exception as e:
            try:
                log_error(city)
                log_error(e)
                log_error(traceback.format_exc())
            except:
                pass
    log_file.close()



## Load txt
import pandas as pd
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
import re
from copy import deepcopy
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import unicodedata
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn import tree
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from itertools import product
import json
import pickle
import psutil
import os
import copy


def savejson(data,out):
    with open(out, 'w') as outfile:
        json.dump(data, outfile,indent=4)

def save_dict(path,datadict,mode='pickle'):
    SAVED = False
    if mode == 'pickle':
        with open(path, 'wb') as outfile:
            pickle.dump(datadict, outfile, protocol=pickle.HIGHEST_PROTOCOL)
        SAVED = True
    else:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(datadict, f, ensure_ascii=False, indent=4)
        SAVED = True
    return SAVED

OUTPUT_PATH = 'out'

cities = ['athens', 'barcelona', 'brasilia', 'buenos-aires', 'caracas', 'kyoto', 'los_angeles', 'melbourne', 'mexico_city', 'montreal', 'warsaw',]

city = cities[0]

file_pattern = f'./generated-facts/generated-facts-%city%-%veracity%.txt'

def load_text_file_yorguin(file_path):
    with open(file_path, 'r',encoding='utf8') as file:
        data = file.readlines()

        for l,line in enumerate(data):
            if '[' in line:
                data[l] = line.split('[')[0] # get rid of explanations that i asked in the case of fakes
            data[l] = data[l].strip()

        data = [line for line in data if line]
    return data

dataframes = []
for city in cities:
    for veracity in ['true','subtly fake','obviously fake','grounded truth']:
        data=load_text_file_yorguin(file_pattern.replace('%city%',city).replace('%veracity%',veracity))
        df = pd.DataFrame(data,columns=['fact'])
        df['city'] = city
        df['veracity'] = veracity
        dataframes.append(df)

df_learn = pd.concat(dataframes,ignore_index=True)

shuffledf=lambda df: df.sample(frac=1).reset_index(drop=True)

df_learn = shuffledf(df_learn)

other_cities = ['10cities(50)','10cities(250)','berlin']

def load_text_file_others(file_path):
    with open(file_path, 'r',encoding='utf8') as file:
        data = file.read()
    return data.splitlines()

dataframes_others = []
for city in other_cities:
    for veracity in ['fakes','facts']:
        file_pattern = f'other_datasets/%veracity%-%city%.txt'
        data=load_text_file_others(file_pattern.replace('%city%',city).replace('%veracity%',veracity))
        df = pd.DataFrame(data,columns=['fact'])
        df['city'] = city
        df['veracity'] = veracity
        dataframes_others.append(df)

df_others = pd.concat(dataframes_others,ignore_index=True)

df_others = shuffledf(df_others)

df_others['source']='others'
df_learn['source']='yorguin'

#df=pd.concat([df_learn,df_others],ignore_index=True)
#df.to_csv('df.csv')
#df[df['veracity'].isin(['true','subtly fake','obviously fake'])]['veracity'].value_counts()

## Preprocessing for one sentence to check behavior of the pipeline

example_sentence = df_others['fact'].iloc[0]

# case conversion
case_conversion = lambda x: x.lower()
example_sentence = case_conversion(example_sentence)

# remove punctuation
remove_punctuation = lambda x: re.sub(r'[^a-zA-Z0-9\s]', ' ', x)
example_sentence = remove_punctuation(example_sentence)

# remove accents
def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])
remove_accents_foo = lambda x: remove_accents(x)
example_sentence = remove_accents_foo(example_sentence)

# tokenize
# word_tokenize is a function in the nltk library that uses the recommended word tokenizer.
example_sentence = word_tokenize(example_sentence)

# remove non-alpha words  [to do or not to do?]
remove_non_alpha = lambda x: [word for word in x if word.isalpha()]
example_sentence = remove_non_alpha(example_sentence)

# remove stopwords [to do or not to do?]
stop_words = set(stopwords.words('english'))
remove_stopwords = lambda x: [word for word in x if word not in stop_words]
example_sentence = remove_stopwords(example_sentence)

# stemming or lemmatization

# stemming
stemmer = SnowballStemmer('english')
stem_foo = lambda x: [stemmer.stem(word) for word in x]
stemmed_sentence = stem_foo(example_sentence)

# lemmatization with pos tagging

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""

    # First letter of POS_TAG assigned in upper case

    tag = nltk.pos_tag([word])[0][1][0].upper()

    # first [0] is the first token received by pos_tag
    # [1] is the pos_tag
    # last [0] is the first character of the pos-tag

    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN) # return wordnet.NOUN if no match


lemmatizer = WordNetLemmatizer()
lemmatize_foo = lambda x: [lemmatizer.lemmatize(word,get_wordnet_pos(word)) for word in x]
lemmatized_sentence = lemmatize_foo(example_sentence)

# backbone of the preprocessing pipeline
def prep(sentence, pipeline=[case_conversion, remove_punctuation, remove_accents_foo,word_tokenize, remove_non_alpha, remove_stopwords, stem_foo, lemmatize_foo]):

    for foo in pipeline:
        sentence = foo(sentence)
    return sentence



df_others['fact-prep'] = df_others['fact'].apply(prep)
df_others['label']=df_others['veracity'].apply(lambda x: True if x=='facts' else False)


## Hyperparameter tuning for the pipeline using df_others

## Encoding to vector space
count_vec_params = dict(input='content',
                        encoding='utf-8',
                        decode_error='strict',
                        ngram_range = (1, 1), # unigrams, test bigrams?
                        strip_accents = None, # already done
                        lowercase = False, # already done
                        preprocessor = None, # already done
                        stop_words = None, # already done
                        tokenizer = lambda x: x, # already done
                        token_pattern=None, # already done
                        analyzer = 'word', # already done
                        max_df = 1., # keep all
                        min_df = 0., # keep all
                        vocabulary=None, # to be fit
                        binary = False, # TODO:hmmm... to be decided
                        dtype=np.int64,
                        max_features=100, # hmmm... to be decided or hyperparameter to be tuned?)
                        )

tfidf_vec_params = dict(input='content',
                    encoding='utf-8',
                    decode_error='strict',
                    strip_accents=None, # already done
                    lowercase=False, # already done
                    preprocessor=None, # already done
                    tokenizer=lambda x: x, # already done
                    analyzer='word',
                    stop_words=None, # already done
                    token_pattern=None, # already done
                    ngram_range=(1, 1), # unigrams, test bigrams?
                    max_df=1.0, min_df=1, # keep all
                    max_features=None, # hmmm... to be decided or hyperparameter to be tuned?
                    vocabulary=None, # to be fit
                    binary=False, # TODO:hmmm... to be decided
                    dtype=np.float64,
                    norm='l2', # normalize
                    use_idf=True, # use idf or not
                    smooth_idf=True, # this helps with zero division
                    sublinear_tf=False) # use log normalization
tfidf_no_idf_vec = TfidfVectorizer(use_idf=False)
tfidf_no_norm_vec = TfidfVectorizer(use_idf=False, norm=None)

gram_vectorizerH = TfidfVectorizer(**tfidf_vec_params) #CountVectorizer(**count_vec_params)



ih_train, ih_test, _, _ = train_test_split(df_others.index, df_others.index, test_size=0.33, shuffle=True, random_state=42, stratify=df_others['label'])


Xh_train = gram_vectorizerH.fit_transform(df_others['fact-prep'].iloc[ih_train])
Xh_test = gram_vectorizerH.transform(df_others['fact-prep'].iloc[ih_test])
yh_train = df_others['label'].iloc[ih_train]
yh_test = df_others['label'].iloc[ih_test]

gram_vectorizerH.vocabulary_

CLASS_WEIGHT={0:1,1:1}



internal_njobs = psutil.cpu_count(logical=False)
model_zoo= {
    'LogisticRegression':{
        'init':dict(penalty='l2',
        dual=False,
        tol=0.0001,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=CLASS_WEIGHT,
        random_state=0,
        solver='lbfgs',
        max_iter=300,
        multi_class='auto',
        verbose=0,
        warm_start=False,
        n_jobs=internal_njobs,
        l1_ratio=None),
        'foo':LogisticRegression,
        },
    'DecisionTree':{
        'init': dict(criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        class_weight=CLASS_WEIGHT,#'balanced',#,{0:1,1:5},#
        ccp_alpha=0.0008),
        'foo':tree.DecisionTreeClassifier},
    'xGB':{'init':dict(
    loss='log_loss', 
    learning_rate=0.1, 
    n_estimators=100, 
    subsample=1.0, 
    criterion='friedman_mse', 
    min_samples_split=2, 
    min_samples_leaf=1, 
    min_weight_fraction_leaf=0.0, 
    max_depth=3, 
    min_impurity_decrease=0.0, 
    init=None, 
    random_state=0, 
    max_features=None, 
    verbose=0, 
    max_leaf_nodes=None, 
    warm_start=False, 
    validation_fraction=0.1, 
    n_iter_no_change=None, 
    tol=0.0001, 
    ccp_alpha=0.01),
    'foo':GradientBoostingClassifier
    },
    'SVM':{
        'init':dict(C=1.0, 
                kernel='rbf', 
                degree=3, 
                gamma='scale', 
                coef0=0.0, 
                shrinking=True, 
                probability=True,
                tol=0.001, 
                cache_size=200, 
                class_weight=CLASS_WEIGHT,#None, 
                verbose=False, 
                max_iter=-1, 
                decision_function_shape='ovr', 
                break_ties=False, 
                random_state=0),
        'foo':SVC
    },
    'LinearSVC':{
        'init':dict(
            penalty='l2',
            loss='squared_hinge',
            dual='auto',
            tol=0.0001,
            C=1.0,
            multi_class='ovr',
            fit_intercept=True,
            intercept_scaling=1,
            class_weight=None,
            verbose=0,
            random_state=42,
            max_iter=1000),
        'foo':LinearSVC
    },
    'BernoulliNB':{
        'init':dict(
        alpha=1.0, 
        force_alpha=True,
        binarize=0.0,
        fit_prior=True,
        class_prior=None
    ),
    'foo':BernoulliNB
    }

}


class_weigths=[1] # vary class weight of True class if you want

## hyperparameter selection
param_grid = {}
param_grid['LogisticRegression'] = {
            'penalty': ['l2','l1']
            ,'C': [0.001,0.01, 0.1, 1]#, 10, 100]
            ,'fit_intercept': [True]#,False]
            ,'solver': ['saga']
            ,'class_weight': [{0: 1, 1: X} for X in class_weigths]
            ,'tol': [1e-3,1e-2],#1e-4, 
            }

param_grid['DecisionTree'] = {
    'criterion': ['gini']#, 'entropy']
    ,'splitter': ['best']#,'random']
    ,'max_depth': [5,10, 20, 30]#, 40, 50]
    ,'min_samples_split': [2,5, 10]
    ,'min_samples_leaf': [1, 2, 4]
    ,'ccp_alpha': [0.0,0.001, 0.01, 0.1]
    ,'class_weight': [{0: 1, 1: X} for X in class_weigths]
}

param_grid['SVM'] = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'degree': [2, 3, 4],
    'gamma': ['scale', 'auto'],
    'shrinking': [True, False],
    'tol': [1e-3, 1e-2],
    'class_weight':[{0: 1, 1: X} for X in class_weigths],
}

param_grid['LinearSVC'] = dict(
    penalty=['l2','l1'],
    loss=['hinge','squared_hinge'],
    dual=['auto'],
    tol=[0.0001,0.001,0.01],
    C=[0.01,0.1,1,10,100],
    multi_class=['ovr'],
    fit_intercept=[True,False],
    intercept_scaling=[1],
    class_weight=[{0: 1, 1: X} for X in class_weigths],
    verbose=[0],
    random_state=[42],
    max_iter=[100,1000,10000]
)

param_grid['BernoulliNB'] = {
        'alpha' : [0.1, 0.5, 1.0, 2.0, 10.0], 
        'force_alpha' : [True,False],
        'binarize' : [0.0, 0.5, 1.0, 1.5, 2.0],
        'fit_prior':[True,False],
        'class_prior' : [None]
}

DO_MODEL = ['LogisticRegression','LinearSVC','BernoulliNB']#,'DecisionTree']#['LogisticRegression']#,'DecisionTree','SVM']#['LogisticRegression']#,[['LogisticRegression','xGB','DecisionTree','SVMln']]]#[['SVM','xGB','DecisionTree','SVMln']]#[['xGB','SVM','SVMln','DecisionTree'],['LogisticRegression','xGB','SVM','DecisionTree'],['SVMln','DecisionTree','LogisticRegression','xGB']]#,['LogisticRegression','DecisionTree','SVM']]
# delete the others
delete_models = [x for x in model_zoo.keys() if x not in DO_MODEL]
for dm in delete_models:
    del model_zoo[dm]

def foo_FPR(y_true,y_pred):
    tn, fp, fn, tp  = confusion_matrix(y_true, y_pred,normalize=None,labels=[0,1]).ravel()
    return 100*fp/(fp+tn) if (fp+tn) != 0 else 0

def foo_PPV(y_true,y_pred):
    tn, fp, fn, tp  = confusion_matrix(y_true, y_pred,normalize=None,labels=[0,1]).ravel()
    return 100*tp/(tp+fp) if (tp+fp) != 0 else 0

def foo_FNR(y_true,y_pred):
    tn, fp, fn, tp  = confusion_matrix(y_true, y_pred,normalize=None,labels=[0,1]).ravel()
    return 100*fn/(fn+tp) if (fn+tp) != 0 else 0

def foo_numTrue(y_true,y_pred):
    tn, fp, fn, tp  = confusion_matrix(y_true, y_pred,normalize=None,labels=[0,1]).ravel()
    return 100*(tp+fp)/(tn+fn+fp+fn)


def confusion_matrix_scorer(clf, X, y):
    y_pred = clf.predict(X)
    # foo_FNR(y, y_pred)
    # foo_FPR(y, y_pred)
    # foo_PPV(y, y_pred)#*int(foo_numTrue(y, y_pred)>0)
    
    return {'PPV': foo_PPV(y, y_pred)}


groups_ = df_learn['city']
groups_v = df_others['city'].iloc[ih_train]

for city in df_learn['city'].unique():
    print(city)
    print(df_learn[df_learn['city']==city]['veracity'].value_counts())

sgkf = StratifiedGroupKFold(n_splits=len(np.unique(groups_)), random_state=0, shuffle=True)
sgkf_v = StratifiedGroupKFold(n_splits=len(np.unique(groups_v)), random_state=0, shuffle=True)
X_v = Xh_train
y_v = yh_train

for mod in model_zoo.keys():
    model_root = os.path.join(OUTPUT_PATH,f"model-{mod}_ML-{'hyperparam'}_task-{'TvsF'}_cv-{'split'}")#f"")
    os.makedirs(model_root,exist_ok=True)
    params=deepcopy(model_zoo[mod])
    print('HYPERPARAM')
    print(set(groups_v))
    if not mod in ['LogisticRegression']:
        grid_search = RandomizedSearchCV(params['foo'](**params['init']), param_grid[mod],n_iter=100, random_state=0,cv=sgkf_v.split(X_v, y_v, groups_v), scoring=confusion_matrix_scorer, verbose=3, n_jobs=internal_njobs,refit='PPV')
    else:
        grid_search = GridSearchCV(params['foo'](**params['init']), param_grid[mod],cv=sgkf_v.split(X_v, y_v, groups_v), scoring=confusion_matrix_scorer, verbose=3, n_jobs=internal_njobs,refit='PPV')
    grid_search.fit(X_v, y_v)
    hyperdf=pd.DataFrame(grid_search.cv_results_)
    hyperdf.to_csv(os.path.join(model_root,f'hyperparams.csv'))
    # Get best parameters and best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"Best Parameters: {best_params}")
    print(f"Best Score: {best_score}")
    params['init'].update(best_params)
    #save_dict(os.path.join(model_root,f'grid_search.pickle'),dict(grid_search=grid_search)) # cannot pickle a generator
    save_dict(os.path.join(model_root,f'best_params.pickle'),best_params)
    try:
        save_dict(os.path.join(model_root,f'best_score.json'),best_score,mode='json')
        save_dict(os.path.join(model_root,f'best_params.json'),best_params,mode='json')
    except:
        pass
    model_ = Pipeline([
            #('scale', StandardScaler()),
            #('selection',SelectKBest(k=10)),
            #('selection',SelectFwe()),
            #('clf',GradientBoostingClassifier())]) 
            # ('clf',LogisticRegression())])
            ('clf',params['foo'](**params['init']))])
    

    models=[]
    train_accs={}
    cmtrain_persub={}
    cmtest_persub={}
    Y_test_ans=[]
    Y_test_preds=[]
    Y_test_subs=[]
    Y_test_spaces=[]
    X_test=[]
    for task in ['TvsSF','TvsOF','GTvsSF','GTvsOF']:

        veracity_set=['true','subtly fake'] if task == 'TvsSF' else ['true','obviously fake'] if task == 'TvsOF' else ['grounded truth','subtly fake'] if task == 'GTvsSF' else ['grounded truth','obviously fake']
        thisdf = df_learn[df_learn['veracity'].isin(veracity_set)]
        model_root = os.path.join(OUTPUT_PATH,f"model-{mod}_ML-{'hyperparam'}_task-{task}_cv-{'split'}")
        for i, (train_index, test_index) in enumerate(sgkf.split(df_learn.index, df_learn.index, df_learn['city'])):
            print(f"Fold {i}:")
            #print(f"  Train: index={train_index}")
            print(f"  Train: group={set(groups_[train_index])}")
            #print(f"  Test:  index={test_index}")
            print(f"  Test:  group={set(groups_[test_index])}")
            model = copy.deepcopy(model_)
            #Train the model
            X_ = gram_vectorizerH.transform(df_learn['fact-prep'].iloc[train_index])
            y_ = df_learn['veracity'].apply() .iloc[train_index]
            model.fit(X_.iloc[train_index], y_[train_index]) #Training the model
            yy_true = y_[train_index]
            yy_pred = model.predict(X_.iloc[train_index])
            yyy_true = y_[test_index]
            yyy_pred = model.predict(X_.iloc[test_index])
            Y_test_ans+=yyy_true.tolist()
            Y_test_preds+=yyy_pred.tolist()
            Y_test_subs+=groups_[test_index].tolist()
            Y_test_spaces+=spaces_[test_index].tolist()
            X_test+=X_.iloc[test_index].to_numpy().tolist()

            trainscore = ACCURACY_FOO(yy_true, yy_pred)
            testscore =  ACCURACY_FOO(yyy_true, yyy_pred)
            trainFPR = foo_FPR(yy_true, yy_pred)
            testFPR =  foo_FPR(yyy_true, yyy_pred)
            trainPPV = foo_PPV(yy_true, yy_pred)
            testPPV=  foo_PPV(yyy_true, yyy_pred)
            trainFNR = foo_FNR(yy_true, yy_pred)
            testFNR =  foo_FNR(yyy_true, yyy_pred)

            traincm  = confusion_matrix(yy_true, yy_pred,normalize=None,labels=[0,1])
            testcm  = confusion_matrix(yyy_true, yyy_pred,normalize=None,labels=[0,1])
            subcurr=list(set(groups_[test_index]))
            subcurr.sort()
            sub = ','.join(subcurr)
            cmtrain_persub[sub]={}
            cmtest_persub[sub]={}
            # No run merge here
            cmtrain_persub[sub]['TN'],cmtrain_persub[sub]['FP'],cmtrain_persub[sub]['FN'],cmtrain_persub[sub]['TP']=tuple(traincm.flatten().tolist())
            cmtest_persub[sub]['TN'],cmtest_persub[sub]['FP'],cmtest_persub[sub]['FN'],cmtest_persub[sub]['TP']=tuple(testcm.flatten().tolist())

            print(f"Accuracy for the fold no. {i} on the train set: BACC {trainscore} FPR {trainFPR} PPV {trainPPV}")
            i += 1
            models.append((model,trainscore,testscore,trainFPR,testFPR,trainPPV,testPPV,trainFNR,testFNR,set(groups_[test_index])))
            ### Train Accuracies per set inside the train set.
            fold_str=".".join(list(set(groups_[test_index])))
            train_accs[fold_str]={}
            for gg in set(groups_[train_index]):
                idxs = np.where(groups_[train_index]==gg)[0]
                trainsubbacc =ACCURACY_FOO(y_[train_index][idxs], model.predict(X_.iloc[train_index].iloc[idxs]))
                trainsubfpr =foo_FPR(y_[train_index][idxs], model.predict(X_.iloc[train_index].iloc[idxs]))
                trainsubppv =foo_PPV(y_[train_index][idxs], model.predict(X_.iloc[train_index].iloc[idxs]))
                trainsubfnr =foo_FNR(y_[train_index][idxs], model.predict(X_.iloc[train_index].iloc[idxs]))
                print(f"Accuracy for {gg} on the train set: BACC {trainsubbacc} FPR {trainsubfpr} PPV {trainsubppv} FNR {trainsubfnr}")
                train_accs[fold_str][gg]={"score":trainsubbacc,"fpr":trainsubfpr,"ppv":trainsubppv,'fnr':trainsubfnr,"n":len(idxs),"percentage":len(idxs)/len(train_index),"total":len(train_index),"fold":i}

            print(f"Accuracy for the fold no. {i} on the test set: BACC {testscore} FPR {testFPR} PPV {testPPV} FNR {testFNR}")

        savejson(train_accs,os.path.join(model_root,f"trainaccuracies.json"))
        savejson(cmtrain_persub,os.path.join(model_root,f"cmtrain.json"))
        savejson(cmtest_persub,os.path.join(model_root,f"cmtest.json"))

"""
clf = LogisticRegression(random_state=0).fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)


confusion_matrix(y_test, y_pred)

# feature importance
# plot the top 10 features and their coefficients


def plot_coefficients(classifier, feature_names, top_features=10):
    coef = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure(figsize=(15, 5))
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
    plt.show()

plot_coefficients(clf, np.array(gram_vectorizer.get_feature_names_out()))
"""