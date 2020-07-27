from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')
def index():
    menu = {'home':True, 'rgrs':False, 'stmt':False, 'clsf':False, 'clst':False, 'user':False}
    return render_template('home.html', menu=menu)

@app.route('/regression', methods=['GET', 'POST'])
def regression():
    menu = {'home':False, 'rgrs':True, 'stmt':False, 'clsf':False, 'clst':False, 'user':False}
    if request.method == 'GET':
        return render_template('regression.html', menu=menu)
    else:
        slen = float(request.form['slen'])
        plen = float(request.form['plen'])
        pwid = float(request.form['pwid'])
        species = int(request.form['species'])
        swid = 0.63711424 * slen - 0.535485016 * plen + 0.55807335 * pwid - 0.12647156 * species + 0.78264901
        names = ['Setosa', 'Versicolor', 'Virginica']
        species = names[species]
        iris = {'slen':slen, 'swid':round(swid, 4), 'plen':plen, 'pwid':pwid, 'species':species}
        return render_template('reg_result.html', menu=menu, iris=iris)

@app.route('/sentiment')
def sentiment():
    pass

@app.route('/classification')
def classification():
    pass

@app.route('/clustering')
def clustering():
    pass

if __name__ == '__main__':
    app.run(debug=True) ## deburg = true : 개발모드
