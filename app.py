import aaransia
import pandas as pd
from aaransia import transliterate, SourceLanguageError
import re
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import logging

# Load train
df = pd.read_csv('train.csv')
temp = pd.read_csv('dev.csv')
#merge train and dev
df = pd.concat([df, temp], ignore_index=True)
df['tweet'] = df['tweet'].astype(str) # One row has a float as the text
print(df.shape)
#df.head()

EMOTICONS = {
    u":)":"Happy face or smiley",u":D":"Happy face or smiley",    u"<3":"Happy face or smiley",    u":*":"Happy face or smiley",    u":p":"Happy face or smiley",    u":P":"Happy face or smiley",    u"xD":"Happy face or smiley",    u"XD":"Happy face or smiley",    u":â€‘\)":"Happy face or smiley",    u":\)":"Happy face or smiley",    u":-\]":"Happy face or smiley",    u":\]":"Happy face or smiley",    u":-3":"Happy face smiley",    u":3":"Happy face smiley",    u":->":"Happy face smiley",    u":>":"Happy face smiley",    u"8-\)":"Happy face smiley",    u":o\)":"Happy face smiley",    u":-\}":"Happy face smiley",    u":\}":"Happy face smiley",    u":-\)":"Happy face smiley",    u":c\)":"Happy face smiley",    u":\^\)":"Happy face smiley",    u"=\]":"Happy face smiley",
    u"=\)":"Happy face smiley",    u":â€‘D":"Laughing, big grin or laugh with glasses",    u":D":"Laughing, big grin or laugh with glasses",    u"8â€‘D":"Laughing, big grin or laugh with glasses",    u"8D":"Laughing, big grin or laugh with glasses",    u"Xâ€‘D":"Laughing, big grin or laugh with glasses",    u"XD":"Laughing, big grin or laugh with glasses",    u"=D":"Laughing, big grin or laugh with glasses",    u"=3":"Laughing, big grin or laugh with glasses",    u"B\^D":"Laughing, big grin or laugh with glasses",    u":-\)\)":"Very happy",    u":â€‘\(":"Frown, sad, andry or pouting",    u":-\(":"Frown, sad, andry or pouting",    u":\(":"Frown, sad, andry or pouting",    u":â€‘c":"Frown, sad, andry or pouting",   u":c":"Frown, sad, andry or pouting",   u":â€‘<":"Frown, sad, andry or pouting",    u":<":"Frown, sad, andry or pouting",    u":â€‘\[":"Frown, sad, andry or pouting",    u":\[":"Frown, sad, andry or pouting",
    u":-\|\|":"Frown, sad, andry or pouting",    u">:\[":"Frown, sad, andry or pouting",  u":\{":"Frown, sad, andry or pouting",    u":@":"Frown, sad, andry or pouting",    u">:\(":"Frown, sad, andry or pouting",    u":'â€‘\(":"Crying",    u":'\(":"Crying",    u":'â€‘\)":"Tears of happiness",    u":'\)":"Tears of happiness",    u"Dâ€‘':":"Horror",    u"D:<":"Disgust",   u"D:":"Sadness",    u"D8":"Great dismay",    u"D;":"Great dismay",    u"D=":"Great dismay",    u"DX":"Great dismay",    u":â€‘O":"Surprise",   u":O":"Surprise",    u":â€‘o":"Surprise",    u":o":"Surprise",    u":-0":"Shock",    u"8â€‘0":"Yawn",    u">:O":"Yawn",
    u":-\*":"Kiss",    u":\*":"Kiss",    u":X":"Kiss",    u";â€‘\)":"Wink or smirk",    u";\)":"Wink or smirk",    u"\*-\)":"Wink or smirk",    u"\*\)":"Wink or smirk",    u";â€‘\]":"Wink or smirk",    u";\]":"Wink or smirk",    u";\^\)":"Wink or smirk",    u":â€‘,":"Wink or smirk",   u";D":"Wink or smirk",   u":â€‘P":"Tongue sticking out, cheeky, playful or blowing a raspberry",    u":P":"Tongue sticking out, cheeky, playful or blowing a raspberry",  u"Xâ€‘P":"Tongue sticking out, cheeky, playful or blowing a raspberry",    u"XP":"Tongue sticking out, cheeky, playful or blowing a raspberry",    u":â€‘Ãž":"Tongue sticking out, cheeky, playful or blowing a raspberry",    u":Ãž":"Tongue sticking out, cheeky, playful or blowing a raspberry",    u":b":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u"d:":"Tongue sticking out, cheeky, playful or blowing a raspberry",    u"=p":"Tongue sticking out, cheeky, playful or blowing a raspberry",    u">:P":"Tongue sticking out, cheeky, playful or blowing a raspberry",    u":â€‘/":"Skeptical, annoyed, undecided, uneasy or hesitant",    u":/":"Skeptical, annoyed, undecided, uneasy or hesitant",    u":-[.]":"Skeptical, annoyed, undecided, uneasy or hesitant",    u">:[(\\\)]":"Skeptical, annoyed, undecided, uneasy or hesitant",    u">:/":"Skeptical, annoyed, undecided, uneasy or hesitant",    u":[(\\\)]":"Skeptical, annoyed, undecided, uneasy or hesitant",    u"=/":"Skeptical, annoyed, undecided, uneasy or hesitant",    u"=[(\\\)]":"Skeptical, annoyed, undecided, uneasy or hesitant",    u":L":"Skeptical, annoyed, undecided, uneasy or hesitant",    u"=L":"Skeptical, annoyed, undecided, uneasy or hesitant",    u":S":"Skeptical, annoyed, undecided, uneasy or hesitant",    u":â€‘\|":"Straight face",    u":\|":"Straight face",    u":$":"Embarrassed or blushing",
    u":â€‘x":"Sealed lips or wearing braces or tongue-tied",    u":x":"Sealed lips or wearing braces or tongue-tied",    u":â€‘#":"Sealed lips or wearing braces or tongue-tied",    u":#":"Sealed lips or wearing braces or tongue-tied",    u":â€‘&":"Sealed lips or wearing braces or tongue-tied",    u":&":"Sealed lips or wearing braces or tongue-tied",    u"O:â€‘\)":"Angel, saint or innocent",    u"O:\)":"Angel, saint or innocent",    u"0:â€‘3":"Angel, saint or innocent",    u"0:3":"Angel, saint or innocent",    u"0:â€‘\)":"Angel, saint or innocent",    u"0:\)":"Angel, saint or innocent",    u":â€‘b":"Tongue sticking out, cheeky, playful or blowing a raspberry",    u"0;\^\)":"Angel, saint or innocent",    u">:â€‘\)":"Evil or devilish",    u">:\)":"Evil or devilish",    u"\}:â€‘\)":"Evil or devilish",    u"\}:\)":"Evil or devilish",    u"3:â€‘\)":"Evil or devilish",    u"3:\)":"Evil or devilish",    u">;\)":"Evil or devilish",    u"\|;â€‘\)":"Cool",    u"\|â€‘O":"Bored",    u":â€‘J":"Tongue-in-cheek",    u"#â€‘\)":"Party all night",
    u"%â€‘\)":"Drunk or confused",    u"%\)":"Drunk or confused",    u":-###..":"Being sick",    u":###..":"Being sick",    u"<:â€‘\|":"Dump",    u"\(>_<\)":"Troubled",    u"\(>_<\)>":"Troubled",    u"\(';'\)":"Baby",    u"\(\^\^>``":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",    u"\(\^_\^;\)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",    u"\(-_-;\)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",    u"\(~_~;\) \(ãƒ»\.ãƒ»;\)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",    u"\(-_-\)zzz":"Sleeping",    u"\(\^_-\)":"Wink",    u"\(\(\+_\+\)\)":"Confused",    u"\(\+o\+\)":"Confused",    u"\(o\|o\)":"Ultraman",    u"\^_\^":"Joyful",
    u"\(\^_\^\)/":"Joyful",    u"\(\^O\^\)ï¼":"Joyful",    u"\(\^o\^\)ï¼":"Joyful",
    u"\(__\)":"Kowtow as a sign of respect, or dogeza for apology",    u"_\(\._\.\)_":"Kowtow as a sign of respect, or dogeza for apology",    u"<\(_ _\)>":"Kowtow as a sign of respect, or dogeza for apology",    u"<m\(__\)m>":"Kowtow as a sign of respect, or dogeza for apology",    u"m\(__\)m":"Kowtow as a sign of respect, or dogeza for apology",    u"m\(_ _\)m":"Kowtow as a sign of respect, or dogeza for apology",    u"\('_'\)":"Sad or Crying",    u"\(/_;\)":"Sad or Crying",    u"\(T_T\) \(;_;\)":"Sad or Crying",    u"\(;_;":"Sad of Crying",    u"\(;_:\)":"Sad or Crying",    u"\(;O;\)":"Sad or Crying",    u"\(:_;\)":"Sad or Crying",    u"\(ToT\)":"Sad or Crying",    u";_;":"Sad or Crying",    u";-;":"Sad or Crying",    u";n;":"Sad or Crying",    u";;":"Sad or Crying",    u"Q\.Q":"Sad or Crying",    u"T\.T":"Sad or Crying",    u"QQ":"Sad or Crying",    u"Q_Q":"Sad or Crying",
    u"\(-\.-\)":"Shame",    u"\(-_-\)":"Shame",    u"\(ä¸€ä¸€\)":"Shame",    u"\(ï¼›ä¸€_ä¸€\)":"Shame",    u"\(=_=\)":"Tired",    u"\(=\^\Â·\^=\)":"cat",    u"\(=\^\Â·\Â·\^=\)":"cat",    u"=_\^= ":"cat",    u"\(\.\.\)":"Looking down",    u"\(\._\.\)":"Looking down",    u"\^m\^":"Giggling with hand covering mouth",    u"\(\ãƒ»\ãƒ»?":"Confusion",    u"\(?_?\)":"Confusion",    u">\^_\^<":"Normal Laugh",    u"<\^!\^>":"Normal Laugh",    u"\^/\^":"Normal Laugh",    u"\ï¼ˆ\*\^_\^\*ï¼‰" :"Normal Laugh",   u"\(\^<\^\) \(\^\.\^\)":"Normal Laugh",    u"\(^\^\)":"Normal Laugh",   u"\(\^\.\^\)":"Normal Laugh",    u"\(\^_\^\.\)":"Normal Laugh",    u"\(\^_\^\)":"Normal Laugh",    u"\(\^\^\)":"Normal Laugh",    u"\(\^J\^\)":"Normal Laugh",
    u"\(\*\^\.\^\*\)":"Normal Laugh",    u"\(\^â€”\^\ï¼‰":"Normal Laugh",    u"\(#\^\.\^#\)":"Normal Laugh",   u"\ï¼ˆ\^â€”\^\ï¼‰":"Waving",   u"\(;_;\)/~~~":"Waving",    u"\(\^\.\^\)/~~~":"Waving",    u"\(-_-\)/~~~ \($\Â·\Â·\)/~~~":"Waving",    u"\(T_T\)/~~~":"Waving",    u"\(ToT\)/~~~":"Waving",    u"\(\*\^0\^\*\)":"Excited",    u"\(\*_\*\)":"Amazed",
    u"\(\*_\*;":"Amazed",    u"\(\+_\+\) \(@_@\)":"Amazed",    u"\(\*\^\^\)v":"Laughing,Cheerful",    u"\(\^_\^\)v":"Laughing,Cheerful",    u"\(\(d[-_-]b\)\)":"Headphones,Listening to music",    u'\(-"-\)':"Worried",    u"\(ãƒ¼ãƒ¼;\)":"Worried",    u"\(\^0_0\^\)":"Eyeglasses",    u"\(\ï¼¾ï½–\ï¼¾\)":"Happy",   u"\(\ï¼¾ï½•\ï¼¾\)":"Happy",    u"\(\^\)o\(\^\)":"Happy",    u"\(\^O\^\)":"Happy",    u"\(\^o\^\)":"Happy",    u"\)\^o\^\(":"Happy",    u":O o_O":"Surprised",    u"o_0":"Surprised",    u"o\.O":"Surpised",    u"\(o\.o\)":"Surprised",    u"oO":"Surprised",    u"\(\*ï¿£mï¿£\)":"Dissatisfied", u"\(â€˜A`\)":"Snubbed or Deflated"
    }

emojitext=list(EMOTICONS.keys())

import emoji

def extract_emoji(string):
  decode   = string.encode().decode('utf-8')
  allchars = [str for str in decode]
  List     = [c for c in allchars if c in emoji.EMOJI_DATA]
  words= string.split(' ')
  List2    = [c for c in words if c in emojitext]
  return ' '.join(List+List2)
def remove_emoji(string):
  decode   = string.encode().decode('utf-8')
  allchars = [str for str in decode]
  List     = [c for c in allchars if c in emoji.EMOJI_DATA]
  filtred  = [str for str in decode.split() if not any(i in str for i in List)]
  text = ' '.join(filtred)
  words= text.split(' ')
  List2    = [c for c in words if c not in emojitext]
  return ' '.join(List2)
def removeDuplicates(s):
    if s!='':
            a=s[0]
            for i in s:
                if i != a[-1]:
                    a=a+i
            return  a
    else:
        return ''
def preProcessing(tweet):    
    tweet=re.sub(r"http\S+ | www\S+" , " ", tweet)
    tweet=re.sub(r"#" , "", tweet)
    tweet=re.sub(r'@\S+',' ',tweet)
    tweet=removeDuplicates(tweet)
    return tweet
import string
def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text
df['preprocessedtext'] = df['tweet'].apply(lambda x: preProcessing(x))
df['emojis']= df['preprocessedtext'].apply(lambda x: extract_emoji(x))
df['noemojitext'] = df['preprocessedtext'].apply(lambda x: remove_emoji(x))
df['arabictext'] = df['noemojitext'].apply(lambda x: transliterate(x, source='tn', target='ar', universal=True))
df['arabictext'] = df['arabictext'].apply(lambda x: remove_punct(x))

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('stopwords')
stop = stopwords.words('arabic')   #all the stop words
def NoStop (txt):
  Q=[]         #checking for stop words
  a=txt.split(" ")
  for i in a:
    if i in stop or len(i)<=2:
       continue
    else:
      Q.append(i)
  return " ".join(Q)
df['NoStopWords'] = df['arabictext'].apply(NoStop)

#Importing libraries
import re
from emot.emo_unicode import UNICODE_EMOJI, EMOTICONS_EMO

def convert_emoticons(text):
    for emot in EMOTICONS_EMO:
        escaped_emot = re.escape(emot)
        text = re.sub(rf'({escaped_emot})', " ".join(EMOTICONS_EMO[emot].replace(",","").split()), text)
    return text

def convert_emojis(text):
    for emot in UNICODE_EMOJI:
        text = text.replace(emot, " ".join(UNICODE_EMOJI[emot].replace(",","").replace(":","").split()))
    return text

df['emojis'] = df['emojis'].apply(lambda x: x.upper())
df['emojis'] = df['emojis'].apply(convert_emoticons)
df['emojis'] = df['emojis'].apply(convert_emojis)

df['emojis']=df['emojis'].apply(lambda x : x.replace('_'," "))

df['final']=df['NoStopWords']+' '+df['emojis']

from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from nltk.tokenize import regexp_tokenize 

def normalizer(sen):
    tokens=regexp_tokenize(sen, "[\w']+") #toknize words 
    return tokens

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=normalizer)),  # strings to token integer counts
    ('classifier', MultinomialNB()) # train on TF-IDF vectors w/ Naive Bayes classifier
])

pipeline.fit(df['final'], df['label'])

pred_labels = pipeline.predict(df['final'])

df2 = pd.read_csv('test.csv')
df2['preprocessedtext'] = df2['tweet'].apply(lambda x: preProcessing(x))
df2['emojis']= df2['preprocessedtext'].apply(lambda x: extract_emoji(x))
df2['noemojitext'] = df2['preprocessedtext'].apply(lambda x: remove_emoji(x))
df2['arabictext'] = df2['noemojitext'].apply(lambda x: transliterate(x, source='tn', target='ar', universal=True))
df2['arabictext'] = df2['arabictext'].apply(lambda x: remove_punct(x))
df2['NoStopWords'] = df2['arabictext'].apply(NoStop)
df2['emojis']=df2['emojis'].apply(lambda x : x.upper())
df2['emojis']=df2['emojis'].apply(convert_emoticons)
df2['emojis']=df2['emojis'].apply(convert_emojis)
df2['emojis']=df2['emojis'].apply(lambda x : x.replace('_'," "))
df2['final']=df2['NoStopWords']+' '+df2['emojis']

for i in range(len(df2['final'])):
    try:
        df2['final'][i] = transliterate(df2['final'][i], source='ar', target='ma')
    except:
        pass

pred_labels= pipeline.predict(df2['final'])

#from sklearn.metrics import classification_report
#print(classification_report(df2['label'], pred_labels))


############################################################################################################

import streamlit as st
from PIL import Image

def main():

    # $ 2. horizontal menu
    options = ["Home", "Classify", "DarijaGPT"]
    icons = ["🏠", "📦", "🤖"]
    selected = st.selectbox(
    "Welcome to The Predictor!",
    options,
    format_func=lambda option: f"{icons[options.index(option)]} {option}",
    index=0
)


######################################################################


    if selected == "Home":
        raw_data = pd.read_csv("/Users/mountasser/Downloads/Sentiments_Analysis_Darija_Tweets-main/train.csv")
        st.sidebar.title("Checkbox this to show the raw data!")
        #st.image(image1, caption='1337')
        if st.sidebar.checkbox("Show a sample of raw data", False):
            st.subheader("Evaluate Moroccan People's Review")
            st.write(raw_data)
        st.sidebar.image('/Users/mountasser/Downloads/Sentiments_Analysis_Darija_Tweets-main/1337ai.png')
        st.sidebar.image('/Users/mountasser/Downloads/Sentiments_Analysis_Darija_Tweets-main/math-maroc.png')
        st.title("Moroccan Review Poll Solution")
        st.markdown("💳 What do Moroccans think through their internet posting?")
        st.markdown("💳 How satisified are Moroccans with your products? Come find out.")
        st.markdown("💳 This is a project made at ThinkAI Hackathon hosted at 1337 by Math&Maroc and 1337AI")
        #image1 = Image.open('/Users/mountasser/Desktop/Data Mining Project/Images/ENSIAS.png')
        #st.image(image1, caption='ENSIAS')


    if selected == "Classify":
        with st.form("my_form"):
            new_review = st.text_input("Kindly enter your Moroccan review: ")

            def Analyze(new_review):
                y = pipeline.predict([new_review])
                return "that is a {} feedback".format(y[0])

            submitted = st.form_submit_button("Classify Sentiment!")

            if submitted:
                new_prediction = Analyze(new_review)
                if new_prediction[0] == 'negative':
                    st.title("Ops, that was not so good after all!")
                elif new_prediction[0] == 'positive':
                    st.title("Yay, that was a positive one <3")
                else:
                    st.title("That was a neutral one :)")
            #st.write("Outside the form")

    if selected == "DarijaGPT":
        with st.form("DarijaGPT"):
            from DarijaGPTfunc import DarijaGPT
            prompt = st.text_input("Interact with GPT in Darija: ")
            submitted = st.form_submit_button("Ask GPT!")

            if submitted:
                output = DarijaGPT(prompt)
                st.title(output)
            #st.write("Outside the form")


if __name__ == '__main__':
    main()
