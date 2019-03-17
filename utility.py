import os,json
from twilio.rest import Client
CONF_PATH = './configs'
def json2param(filename):
    path = os.path.join(CONF_PATH,filename)
    with open(path,'r') as fp:
        param=json.load(fp)
        fp.close()
        return param
def param2json(param,name):
    path = os.path.join(CONF_PATH, name)
    with open(path, 'w') as fp:
        json.dump(param, fp)
        fp.close()
def notify(cell,MSG):


    # Your Account Sid and Auth Token from twilio.com/console
    # DANGER! This is insecure. See http://twil.io/secure
    account_sid = 'ACf87e05d27690b72ad609d560c9e70506'
    auth_token = 'f9d8be566dd9eda6d1fd61d189be93bf'
    client = Client(account_sid, auth_token)

    message = client.messages \
        .create(
        body=MSG,
        from_='+14439633975',
        to=cell
    )
