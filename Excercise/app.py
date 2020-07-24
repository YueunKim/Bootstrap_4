from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('01_bootstrap.html')

@app.route('/typo', methods = ['GET','POST']) ## get은 화면제공 post는 처리
def typo():
        return render_template('03_typography.html')

@app.route('/iris', methods = ['GET','POST'])
def iris():
    if request.method == 'GET':
        return render_template('12_form_iris.html')
    else:
        slen1 = float(request.form['slen']) * 2  #string이므로 float으로 변환
        plen1 = float(request.form['plen']) * 2
        pwid1 = float(request.form['pwid']) * 2
        species1 = int(request.form['species'])
        comment1 = request.form['comment']
        return render_template('12_iris-result.html', slen=slen1, plen=plen1, pwid=pwid1, species=species1, comment=comment1)

@app.route('/project')
def project():
    return render_template('17_templates.html')

@app.route('/hello/')
@app.route('/hello/<name>') ## name을 argument로 받을 수 있음
def hello(name=None): 
    return render_template('hello.html', name=name)

if __name__ == '__main__':
    app.run(debug=True) ## deburg = true : 개발모드

## parameter : 사람과 서버간의 전달하는 값
## 전달하는 방법이 get과 post가 다름