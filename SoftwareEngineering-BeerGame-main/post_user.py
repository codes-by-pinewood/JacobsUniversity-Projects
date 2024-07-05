#!/usr/bin/env python3
# -*- coding: UTF-8 -*-# enable debugging
import json
from establish_connection import StartConnection
import cgitb
import sys
import bcrypt
import mysql.connector.errors
from create_user_session import create_user_session
cgitb.enable()
response_dict = {
    "status": 400,
    "description": "unknown error"
}

if __name__ == '__main__':
    connection = StartConnection()
    # make sure input is valid
    try:
        a = sys.stdin.read()
        if not a:
            print("Status: 400 Bad Request\n")
            print("Content-Type: application/json;charset=utf-8\n\n")
            response_dict['description'] = "no content body given"
            print(json.dumps(response_dict))
            exit(-1)
        user = json.loads(a)
        if 'username' and  'password' and 'email' not in user:
            print("Status: 400 Bad Request\n")
            print("Content-Type: application/json;charset=utf-8\n\n")
            response_dict['description'] = "missing required fields"
            print(json.dumps(response_dict))
        usr_name = str.encode(user['username'])
        usr_pass = str.encode(user['password'])
        usr_email = str.encode(user['email'])
    except json.JSONDecodeError as err:
        print("Status: 400 Bad Request\n")
        print("Content-Type: application/json;charset=utf-8\n\n")
        response_dict['description'] = "error parsing json"
        print(json.dumps(response_dict))
        exit(-1)
    except Exception as err:
        print("Status: 400 Bad Request\n")
        response_dict['description'] = str(err)
        print(json.dumps(response_dict))
        exit(-1)

    permission = 0

    # try to generate new salt
        #bcrypt stores the salt in the hash
    try:
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(usr_pass, salt)
    except Exception as err:
        print("Status: 500\n")
        print("Content-Type: application/json;charset=utf-8\n\n")
        response_dict['description'] = "An error occurred hashing your password, please try to input a new password"
        print(json.dumps(response_dict))

    # try to insert into database
    try:
        cursor = connection.cursor()
        cursor.execute('''INSERT INTO player (name, email, hash, permission, banned) 
        VALUES (%s, %s, %s, %s, %s)''', 
        (usr_name, 
        usr_email, 
        hashed, 
        permission, 
        0))
        
        cursor.execute('SELECT LAST_INSERT_ID()')
        new_u_id = cursor.fetchone()[0]
        # create a new user session
        user_session = create_user_session(new_u_id, cursor)
        response_dict.update(user_session)
        connection.commit()
        print("Status: 201 \n")
        print("Content-Type: application/json;charset=utf-8\n\n")
        response_dict['user_id'] = new_u_id
        print(json.dumps(response_dict))
        
    except mysql.connector.IntegrityError as err:
        print("Status: 422 Unprocessable Entity\n")
        print("Content-Type: application/json;charset=utf-8\n\n")
        response_dict['description'] = "username / email is already in use!"
        print(json.dumps(response_dict))
    except Exception as err:
        print("Status: 400 Bad Request\n")
        print("Content-Type: application/json;charset=utf-8\n\n")
        response_dict['description'] = str(err)
        print(json.dumps(response_dict))

    connection.close()