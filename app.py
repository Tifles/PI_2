import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

#готовим функцию загрузки моделей
@st.cache(allow_output_mutation=True)
def load_model():
    tokenizer_en_ru = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
    model_en_ru = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
    tokenizer_ru_en = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
    model_ru_en = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
    return model_en_ru, model_ru_en, tokenizer_en_ru, tokenizer_ru_en

# Загружаем предварительно обученных моделей
model_en_ru, model_ru_en, tokenizer_en_ru, tokenizer_ru_en = load_model()

#Заголок
st.title('Перевод текста')

#поле для ввода оригинала текста
buf_orig_text = st.text_input("Введите текс для перевода", " Enter the English text here")
status = st.radio("Направление перевода: ", ('EN->RU', 'RU->EN'))
#при нажатии кнопки
if(st.button('Submit')):
    #перевод с русского на англиский
    if (status == 'RU->EN'):
        orig_text = buf_orig_text.title()
        inputs = tokenizer_en_ru(orig_text, return_tensors="pt")
        output = model_en_ru.generate(**inputs, max_new_tokens=100)
        out_text = tokenizer_en_ru.batch_decode(output, skip_special_tokens=True)
    #перевод с англиского на русский
    else:
        orig_text = buf_orig_text.title()
        inputs = tokenizer_ru_en(orig_text, return_tensors="pt")
        output = model_ru_en.generate(**inputs, max_new_tokens=100)
        out_text = tokenizer_ru_en.batch_decode(output, skip_special_tokens=True)
    result = out_text[0]
    st.success(result)



