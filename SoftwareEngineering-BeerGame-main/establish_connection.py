#!/usr/bin/env python3
import json
from handleRequestCookie import handleRequestCookie
def auth_err(err):
    print('Content-Type: text/html;charset=utf-8\n\n')
    print(json.dumps(err))
    exit(-1)


def norm_err(err = None):
    print('Content-Type: text/html;charset=utf-8\n\n')
    if err == None:
        print('''{"status":500,"description":"an unknown error has occurred"}''')
    else:
        print(json.dumps(err))
    exit(-1)

def StartConnection():
    import json
    import mysql.connector
    from mysql.connector import errorcode
    f = open('userconfig.json')
    config = json.load(f)
    try:
        connection = mysql.connector.connect(**config, auth_plugin='mysql_native_password')
        return connection
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print(err)
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print(err)
        else:
            print(err)
        return None

        

if __name__ == '__main__':
    pass
