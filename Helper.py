import re
import nrrd
import requests

access_token = '5307815831:AAEKVzHQFAVWMwXseU9L1yKKtsBq4YPZM8Y'
chat_id = '503932462'

def read_nrrd_file(filepath):
    '''read and load volume'''
    pixelData, header = nrrd.read(filepath)
    return pixelData[:, :, :96]

def normalize(volume):
    min = -1000  # min value of our data : -1000 # It was discovered that min value is -32877.0
    max = 5000  # max value of our data : 5013 # It was discovered that max value is 31691.0
    range = max - min
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / range
    volume = volume.astype("float32")
    return volume

def process_scan(path):
    volume = read_nrrd_file(path)
    volume = normalize(volume)
    return volume

def sorted_alnum(l):
    def convert(text): return int(text) if text.isdigit() else text
    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def telegram_bot_sendtext(bot_message):
    bot_token = access_token
    bot_chatID = chat_id
    send_text = 'https://api.telegram.org/bot' + bot_token + \
        '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + bot_message
    response = requests.get(send_text)
    return response.json()

def send_photo(file_opened):
    api_url = 'https://api.telegram.org/bot' + access_token + '/'
    photo_method = "sendPhoto"
    params = {'chat_id': chat_id}
    files = {'photo': file_opened}
    resp = requests.post(api_url + photo_method, params, files=files)
    return resp