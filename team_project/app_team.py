from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import joblib
import numpy as np
import pandas as pd
import re
from PIL import Image
from konlpy.tag import Okt
from tensorflow import keras
from keras.models import load_model
from keras.applications.vgg16 import VGG16, decode_predictions
from clu_util import cluster_util
from spam_util import spam_mail_util
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB  
from sklearn.metrics import accuracy_score  
from sklearn.model_selection import train_test_split

app = Flask(__name__)
app.debug = True

vgg = VGG16()
okt = Okt()





mail_cv = None
mail_cv_dtm = None
def load_mail_cv():
    global mail_cv, mail_cv_dtm
    mail_cv = joblib.load(os.path.join(app.root_path, 'model/mail_cv.pkl'))
    mail_cv_dtm = joblib.load(os.path.join(app.root_path, 'model/mail_cv_dtm.pkl'))

# mail_tf = None
# mail_tf_dtm = None
# def load_mail_tf():
#     global mail_tf, mail_tf_dtm
#     mail_tf = joblib.load(os.path.join(app.root_path, 'model/mail_tf.pkl'))
#     mail_tf_dtm = joblib.load(os.path.join(app.root_path, 'model/mail_tf_dtm.pkl'))





movie_lr = None
movie_lr_dtm = None
def load_movie_lr():
    global movie_lr, movie_lr_dtm
    movie_lr = joblib.load(os.path.join(app.root_path, 'model/movie_lr.pkl'))
    movie_lr_dtm = joblib.load(os.path.join(app.root_path, 'model/movie_lr_dtm.pkl'))

def tw_tokenizer(text):
    # 입력 인자로 들어온 text 를 형태소 단어로 토큰화 하여 list 객체 반환
    tokens_ko = okt.morphs(text)
    return tokens_ko

movie_nb = None
movie_nb_dtm = None
def load_movie_nb():
    global movie_nb, movie_nb_dtm
    movie_nb = joblib.load(os.path.join(app.root_path, 'model/movie_nb.pkl'))
    movie_nb_dtm = joblib.load(os.path.join(app.root_path, 'model/movie_nb_dtm.pkl'))

def nb_transform(review):
    stopwords=['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
    review = review.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
    morphs = okt.morphs(review, stem=True)
    temp = ' '.join(morph for morph in morphs if not morph in stopwords)
    return temp

model_iris_lr = None
model_iris_svm = None
model_iris_dt = None
model_iris_deep = None 

def load_iris():
    global model_iris_lr, model_iris_svm, model_iris_dt, model_iris_deep
    model_iris_lr = joblib.load(os.path.join(app.root_path, 'model/iris_lr.pkl'))
    model_iris_svm = joblib.load(os.path.join(app.root_path, 'model/iris_svm.pkl'))
    model_iris_dt = joblib.load(os.path.join(app.root_path, 'model/iris_dt.pkl'))
    model_iris_deep = load_model(os.path.join(app.root_path, 'model/iris_deep.hdf5'))

# mail_nb = None
# def load_mail_nb():
#     global mail_nb
#     mail_nb = joblib.load(os.path.join(app.root_path, 'model/mail_nb.pkl'))

@app.route('/')
def index():
    menu = {'home':True, 'rgrs':False, 'stmt':False, 'clsf':False, 'clst':False, 'spam':False, 'user':False}
    return render_template('home.html', menu=menu)



@app.route('/regression_swid', methods=['GET', 'POST'])
def regression_swid():
    menu = {'home':False, 'rgrs':True, 'stmt':False, 'clsf':False, 'clst':False, 'user':False}
    if request.method == 'GET':
        return render_template('regression_swid.html', menu=menu)
    else:
        sp_names = ['Setosa', 'Versicolor', 'Virginica']
        slen = float(request.form['slen'])      # Sepal Length
        plen = float(request.form['plen'])      # Petal Length
        pwid = float(request.form['pwid'])      # Petal Width
        sp = int(request.form['species'])       # Species
        species = sp_names[sp]
        swid = 0.63711424 * slen - 0.53485016 * plen + 0.55807355 * pwid - 0.12647156 * sp + 0.78264901
        swid = round(swid, 4)
        iris = {'slen':slen, 'swid':swid, 'plen':plen, 'pwid':pwid, 'species':species}
        return render_template('reg_result_swid.html', menu=menu, iris=iris)


@app.route('/regression_pwid', methods=['GET', 'POST'])
def regression_pwid():
    menu = {'home':False, 'rgrs':True, 'stmt':False, 'clsf':False, 'clst':False, 'spam':False, 'user':False}
    if request.method == 'GET':
        return render_template('regression_pwid.html', menu=menu)
    else:
        sp_names = ['Setosa', 'Versicolor', 'Virginica']
        slen = float(request.form['slen'])      # Sepal Length
        swid = float(request.form['swid'])      # Petal Width
        plen = float(request.form['plen'])      # Petal Length
        sp = int(request.form['species'])       # Species
        species = sp_names[sp]
        pwid = -0.12783875 * slen + 0.18177682 * swid + 0.32222156 * plen + 0.34709151 * sp -0.16986229175897916
        pwid = round(pwid, 4)
        iris = {'slen':slen, 'swid':swid, 'plen':plen, 'pwid':pwid, 'species':species}
        return render_template('reg_result_pwid.html', menu=menu, iris=iris)


@app.route('/sentiment', methods=['GET', 'POST'])
def sentiment():
    menu = {'home':False, 'rgrs':False, 'stmt':True, 'clsf':False, 'clst':False, 'spam':False, 'user':False}
    if request.method == 'GET':
        return render_template('sentiment.html', menu=menu)
    else:
        res_str = ['부정','긍정']
        review = request.form['review'] # STRING이므로 변환필요없음

        # lr 처리
        review_lr = re.sub(r"\d+", " ", review)
        review_lr_dtm = movie_lr_dtm.transform([review_lr])
        result_lr = res_str[movie_lr.predict(review_lr_dtm)[0]]

        # NB 처리
        review_nb = nb_transform(review)
        review_nb_dtm = movie_nb_dtm.transform([review_nb])
        result_nb = res_str[movie_nb.predict(review_nb_dtm)[0]]

        # 결과처리
        movie = {'review':review, 'result_lr':result_lr, 'result_nb':result_nb}
        return render_template('senti_result.html', menu=menu, movie=movie)

@app.route('/classification', methods=['GET', 'POST'])
def classification():
    menu = {'home':False, 'rgrs':False, 'stmt':False, 'clsf':True, 'clst':False, 'spam':False, 'user':False}
    if request.method == 'GET':
        return render_template('classification.html', menu=menu)
    else:
        f = request.files['image']
        filename = os.path.join(app.root_path, 'static/images/uploads/') + \
                    secure_filename(f.filename)
        f.save(filename)
        img = np.array(Image.open(filename).resize((224, 224)))
        yhat = vgg.predict(img.reshape(-1, 224, 224, 3))
        label_key = np.argmax(yhat)
        label = decode_predictions(yhat)
        label = label[0][0]
        return render_template('cla_result.html', menu=menu, 
                                filename = secure_filename(f.filename),
                                name = label[1], pct = '%.2f'%(label[2] * 100))


@app.route('/classification_iris', methods=['GET', 'POST'])
def classification_iris():
    menu = {'home':False, 'rgrs':False, 'stmt':False, 'clsf':True, 'clst':False, 'spam':False, 'user':False}
    if request.method == 'GET':
        return render_template('classification_iris.html', menu=menu)
    else:
        sp_names = ['Setosa', 'Versicolor', 'Virginica']
        slen = float(request.form['slen'])
        swid = float(request.form['swid'])
        plen = float(request.form['plen'])
        pwid = float(request.form['pwid'])
        test_data = np.array([slen, swid, plen, pwid]).reshape(1,4)
        species_lr = sp_names[model_iris_lr.predict(test_data)[0]]
        species_svm = sp_names[model_iris_svm.predict(test_data)[0]]
        species_dt = sp_names[model_iris_dt.predict(test_data)[0]]
        species_deep = sp_names[model_iris_deep.predict_classes(test_data)[0]]
        iris = {'slen':slen, 'swid':round(swid, 4), 'plen':plen, 'pwid':pwid, 
                'species_lr':species_lr, 'species_svm':species_svm, 
                'species_dt':species_dt, 'species_deep':species_deep}
        return render_template('cla_iris_result.html', menu=menu, iris=iris)        

@app.route('/clustering', methods=['GET', 'POST'])
def clustering():
    menu = {'home':False, 'rgrs':False, 'stmt':False, 'clsf':False, 'clst':True, 'spam':False, 'user':False}
    if request.method=='GET':
        return render_template('clustering.html', menu=menu)
    else:
        f = request.files['csv']
        filename = os.path.join(app.root_path, 'static/images/uploads/') + \
                    secure_filename(f.filename)
        f.save(filename)
        ncls = int(request.form['K'])
        cluster_util(app, ncls, secure_filename(f.filename))
        img_file = os.path.join(app.root_path, 'static/images/kmc.png')
        mtime = int(os.stat(img_file).st_mtime)
        return render_template('clu_result.html', menu=menu, K=ncls, mtime=mtime)


@app.route('/spam', methods=['GET', 'POST'])
def spam():
    menu = {'home':False, 'rgrs':False, 'stmt':False, 'clsf':False, 'clst':False, 'spam':True, 'user':False}
    if request.method=='GET':
        return render_template('spam.html', menu=menu)
    else:
        f = request.files['csv']
        filename = os.path.join(app.root_path, 'data/') + secure_filename(f.filename)
        #f.save(filename)
        # n = request.get_data('acc')
        acc1 = spam_mail_util(app, secure_filename(f.filename))[0]
        acc2 = spam_mail_util(app, secure_filename(f.filename))[1]
        return render_template('spam_result.html', menu=menu, acc1 = round(acc1,4), acc2 = round(acc2,4))

@app.route('/spam_self', methods=['GET', 'POST'])
def spam_self():
    menu = {'home':False, 'rgrs':False, 'stmt':False, 'clsf':False, 'clst':False, 'spam':True, 'user':False}
    if request.method=='GET':
        return render_template('spam_self.html', menu=menu)
    else:
        res_str = ['HAM','SPAM']
        write = request.form['write'] # STRING이므로 변환필요없음

        # dtmvector = CountVectorizer()
        # X_train_dtm = dtmvector.fit_transform(X_train)
        
        # model = MultinomialNB()
        # model.fit(X_train_dtm, y_train)
        # X_test_dtm = dtmvector.transform(X_test)
        # result_cv = model.predict(X_test_dtm)

        # acc1 = accuracy_score(y_test, predicted1)

        # tfidf_transformer = TfidfTransformer()
        # tfidfv = tfidf_transformer.fit_transform(X_train_dtm)   
        # model.fit(tfidfv, y_train)
        # tfidfv_test = tfidf_transformer.fit_transform(X_test_dtm)
        # result_tf = model.predict(tfidfv_test)

        # acc2 = accuracy_score(y_test, predicted2)




        # CV 처리
        # write_cv = re.sub(r"\d+", " ", write)
        write_cv_dtm = mail_cv_dtm.transform([write])
        result_cv = res_str[mail_cv.predict(write_cv_dtm)[0]]

        # TF 처리
        # write_tf = nb_transform(write)
        # x = mail_tf_dtm.transform(['write'])
        # write_tf_dtm = tfidf_transformer.transform(x)   

        # write_tf_dtm = mail_tf_dtm.transform([write])
        # result_tf = res_str[mail_tf.predict(write_tf_dtm)[0]]

        # 결과처리 'result_tf':result_tf
        mail = {'write':write, 'result_cv':result_cv}
        return render_template('spam_self_result.html', menu=menu, mail=mail)


@app.route('/member/<name>')
def member(name):
    menu = {'home':False, 'rgrs':False, 'stmt':False, 'clsf':False, 'clst':False, 'mail':False, 'user':True}
    return render_template('user.html', menu=menu, name=name)


if __name__ == '__main__':
    load_movie_lr()
    load_movie_nb()
    load_mail_cv()
    # load_mail_tf()
    # load_mail_nb()
    load_iris()
    app.run() ## deburg = true : 개발모드 , host='0.0.0.0' = 외부접속허용
