from flask import Flask, render_template, url_for, request, redirect, session
from flask_mysqldb import MySQL
import os

app = Flask(__name__)

app.secret_key = os.urandom(16)

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'password'
app.config['MYSQL_DB'] = 'se_project'

mysql = MySQL(app)

def mysql_connection():
    cursor = mysql.connection.cursor()
    return cursor


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        cur = mysql_connection()
        cur.execute("SELECT * FROM Instructor I WHERE I.i_username = %s AND I.i_password = %s", (username, password))
        account = cur.fetchone()
        if account:
            session['loggedin'] = True
            session['username'] = username
            return render_template("creategame.html")
        else:
            return "Incorrect Username/Password"
    else:
        return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        login_credentials = request.form
        username = login_credentials["username"]
        password = login_credentials['password']
        cur = mysql_connection()
        cur.execute('INSERT INTO Instructor(i_username, i_password) VALUES(%s, %s)', (username, password))
        mysql.connection.commit()
        cur.close()
        return "Account Created"
    else:
        return render_template("register.html")

@app.route('/player_login', methods=['GET', 'POST'])
def player_login():
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        cur = mysql_connection()
        cur.execute("SELECT * FROM Player P WHERE P.p_username = %s AND P.p_password = %s", (username, password))
        account = cur.fetchone()
        if account:
            session['loggedin'] = True
            session['username'] = username
            return 'Logged in successfully!'
        else:
            return "Incorrect Username/Password"
    else:
        return render_template('player_login.html')

if __name__ == "__main__":
    app.run(debug=True)
