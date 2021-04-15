from flask import Flask
from flask import request
import requests
import json
from datetime import datetime
import mysql.connector

app = Flask(__name__)

@app.route("/", methods=['GET'])
def hello_world():
    msg = "Hello IVT-17!"
    if request.args.get('url'):
        ip_client = request.remote_addr
        try:
            r = requests.get(request.args['url'])
            if r.status_code==200:
                return r.content
            else:
                msg = str(r.status_code)
        except:
            msg = f"Не удалось загрузить {request.args.get('url')}"
        try:
            config = json.load(open('/config/config.json'))
            mysql_server = config['ip']
            cnx = mysql.connector.connect(user='myproxy@the-mysql', database='myproxy',
                                          host=mysql_server, password='1234zxcv')
            cursor = cnx.cursor()
            now = datetime.now()
            cursor.execute(
                   "INSERT INTO myproxy.log VALUES(%s, %s, %s)",
                   (now, ip_client, request.args['url'])
            )
            cnx.commit()
        except:
            msg = f"Не удалось записать информацию на сервер базы данных {mysql_server}"
        
    return '<h1>' + msg + '''</h1>
    <form method="get">
    <input type="text" name="url">
    <input type="submit">
    </form>
    '''

if __name__ == '__main__':
    app.run()
