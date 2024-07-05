#!/usr/bin/env python3

import json
from base64 import b64encode
import mysql.connector
from mysql.connector import errorcode
import cgitb
import sys
import bcrypt
import secrets
import datetime
import http.cookies as cookie
debug = True

# Note: Search "--comment--" for new comments that have to be changed.

authentication_response_dict = {
    "status": 400,
    "session_id": None,
    "description": "default"
}
# TODO refactor code
def create_user_session(u_id, cursor):
    user_session_dict = {
        "status": 400,
        "session_id": None,
    }
    token = secrets.token_bytes()
    token = b64encode(token)
    cursor.execute("SELECT s.user_id_fk, s.date, s.session_id from user_session s WHERE s.user_id_fk = %s", (u_id,)) # --comment-- new Table?
    for (user_id, date, session_id) in cursor.fetchall(): #--comment-- cursor fetchall?
        if date + datetime.timedelta(minutes=30) < datetime.datetime.utcnow():
            cursor.execute("DELETE FROM user_session WHERE session_id = %s", (session_id, ))
    #make sure no clashes occur generating a token id, messy but practical solution
    while True:
        try:
            cursor.execute("INSERT INTO user_session (session_id, user_id_fk, date) VALUES (%s, %s, %s)", (token, u_id, datetime.datetime.utcnow()))
            #in case we have a token clash, just try to re-calculate a new token
        except mysql.connector.IntegrityError as err:
            token = b64encode(secrets.token_bytes())
            continue
        except Exception as err:
            user_session_dict['description'] = str(err)
            return user_session_dict
        break
    user_session_dict['session_id'] = token.decode('utf-8')
    user_session_dict['status'] = 201
    user_session_dict['description'] = ""
    respCookie = cookie.SimpleCookie()
    respCookie['se-project'] = token.decode('utf-8') # --comment-- 
    expiration_date = datetime.datetime.utcnow() + datetime.timedelta(minutes=30)
    respCookie['se-project']['expires'] = expiration_date.strftime("%a, %d %b %Y %H:%M:%S UTC") # --comment--
    print("access-control-expose-headers: Set-Cookie")
    respCookie['se-project']['path'] = "/" #--comment--
    print(respCookie)
    return user_session_dict

if __name__ == "__main__":
    cgitb.enable()
    f = open('userconfig.json') #--comment-- JSON File
    config = json.load(f)
    try:
        connection = mysql.connector.connect(**config, auth_plugin='mysql_native_password', charset='utf8') #--comment-- mysql password
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)
    else:
        # print("successfully connected!")
        '''print("content length is : " + os.environ['CONTENT_LENGTH'])
        if int(os.environ['CONTENT_LENGTH']) > 250:
            print("Status: 400 Bad Request\n")
            exit(0)'''

    usr_name = usr_pass = None
    # make sure input is valid
    try:
        a = sys.stdin.read()
        if not a:
            authentication_response_dict['description'] = "no body in GET request"
            #print("Status: 400 Bad Request\n")
            #print("no content body received")
            print("Content-Type: text/html;charset=utf-8\n\n")
            print(json.dumps(authentication_response_dict))
            exit()
        user = json.loads(a)
        usr_name = str.encode(user['username'])
        usr_pass = str.encode(user['password'])
        if not usr_pass or not usr_name:
            #print("Status: 400 Bad Request\n")
            authentication_response_dict['description'] = "no user name / password given"
            print("Content-Type: text/html;charset=utf-8\n\n")
            print(json.dumps(authentication_response_dict))
            exit()
    except json.JSONDecodeError as err:
        #print("Status: 400 Bad Request\n")
        authentication_response_dict['description'] = "Invalid JSON in request aaa"
        print("Content-Type: text/html;charset=utf-8\n\n")
        print(json.dumps(authentication_response_dict))
        #print(err)
        exit()
    except KeyError as err:
        #print("Status: 400 Bad Request\n")
        authentication_response_dict['description'] = "Invalid JSON in request aaaa"
        print("Content-Type: text/html;charset=utf-8\n\n")
        print(json.dumps(authentication_response_dict))
        #print(err)
        exit()

    try:
        cursor = connection.cursor()
        cursor.execute("SELECT u.user_id, u.hash FROM user u where u.name = %s LIMIT 1", (usr_name,)) #--comment-- Hashing, Table, may remove hashing and use password
        usr = cursor.fetchone()
        if usr is None:
            authentication_response_dict['status'] = 401
            authentication_response_dict['description'] = "username not found"
            print("Content-Type: text/html;charset=utf-8\n\n")
            print(json.dumps(authentication_response_dict))
            #print("user not found!")
            #todo implement this
            exit()
        else:
            (u_id, u_hash) = usr
    
            if (isinstance(u_hash, str)):                   
                u_hash = str.encode(''.join(u_hash))
            elif (isinstance(u_hash, bytearray)):
                u_hash = bytes(u_hash)
            # case works: 
            else:
                print("Status: 500 Internal Error\n")
                print("Content-Type: text/html;charset=utf-8\n\n")
                authentication_response_dict['status'] = 500
                authentication_response_dict['description'] = "could not match type of database return"
                print(json.dumps(authentication_response_dict))
                exit()
            if bcrypt.checkpw(usr_pass, u_hash) is True:
                resp = create_user_session(u_id, cursor)
                if (resp['status'] >= 200) and (resp['status'] <= 400):
                    #case: it works
                    print("Content-Type: application/json;charset=utf-8\n\n")
                    
                    authentication_response_dict.update(resp)
                    print(json.dumps(authentication_response_dict))
                    #remember to commit connection 
                    connection.commit()
                    exit(0)
            #case doesn't work:
            else:
                authentication_response_dict['status'] = 401
                authentication_response_dict['description'] = "invalid password"
                print("Content-Type: text/html;charset=utf-8\n\n")
                print(json.dumps(authentication_response_dict))
    except mysql.connector.IntegrityError as err:

        print(err)
        print(err.__doc__)
        print(type(err).__name__)
        exit(-1)
    # except Exception as err:
    #     print(err)
    #     print(err.__doc__)
    #     print(type(err).__name__)
    #     exit(-1)

    connection.close()


