from flask import Flask, render_template, request, redirect, url_for, session
from services.similarity import TextComparer, AuthService
# Декоратор для захисту маршруту
from functools import wraps
import datetime

app = Flask(__name__)
app.secret_key = 'secure-secret-key'

# Сервіс аутентифікації
auth_service = AuthService()


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if auth_service.register(email, password):
            session['user'] = email
            return redirect(url_for('compare'))
        else:
            error = 'Користувач уже існує.'
    return render_template('register.html', error=error)


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if auth_service.authenticate(email, password):
            session['user'] = email
            return redirect(url_for('compare'))
        else:
            error = 'Неправильна електронна пошта або пароль.'
    return render_template('login.html', error=error)


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))


@app.route('/', methods=['GET', 'POST'])
@login_required
def compare():
    result = None
    if request.method == 'POST':
        # Завантаження файлів або форма тексту
        file1 = request.files.get('file1')
        file2 = request.files.get('file2')
        if file1 and file1.filename:
            text1 = file1.read().decode('utf-8', errors='ignore')
        else:
            text1 = request.form.get('text1', '')

        if file2 and file2.filename:
            text2 = file2.read().decode('utf-8', errors='ignore')
        else:
            text2 = request.form.get('text2', '')

        comparer = TextComparer(text1, text2)
        result = comparer.compare_all()
        result['generated'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
