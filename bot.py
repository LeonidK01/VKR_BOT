import asyncio
from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
import joblib
from lem import lem
from keras.utils.data_utils import pad_sequences
import pickle




bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

def predict(text):

    texts = lem(str(text))
    print(texts)
    max_features = 30000
    maxlen = 300

    token_texts = loaded_tokenizer.texts_to_sequences(texts)
    texts_seq = pad_sequences(token_texts, maxlen=maxlen)

    y_pred_pred = model.predict(texts_seq, verbose=1, batch_size=2)
    res_string="\nНормальное сообщение:" + str(round(y_pred_pred[0][0],4)) + "\nоскорбление:" + str(round(y_pred_pred[0][1],4)) + "\nугроза:" + str(round(y_pred_pred[0][2],4)) +"\nнепристойность:" + str(round(y_pred_pred[0][3],4))
    if round(y_pred_pred[0][0],4)<0.5:
        return True,res_string
    else:
        return False,res_string

@dp.message_handler(content_types=types.ContentTypes.TEXT)
async def process_text_message(message: types.Message):
    cheak,text=predict(message.text)
    if cheak:
        await message.delete()
        await bot.send_message(message.chat.id, "Сообщение удалено")
        username = message.from_user.username or message.from_user.first_name
        await bot.send_message(message.chat.id, f"{username}, пожалуйста, соблюдайте уважительное общение в этом чате.{text}")
    else:
        await bot.send_message(message.chat.id,
                               f"{text}")

if __name__ == '__main__':
    model = joblib.load("model.pkl")
    with open('tokenizer.pickle', 'rb') as handle:
        loaded_tokenizer = pickle.load(handle)
    executor.start_polling(dp)
