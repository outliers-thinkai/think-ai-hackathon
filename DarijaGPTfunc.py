import openai
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.getenv('OPENAI_API_KEY')


def get_completion(prompt, model="gpt-3.5-turbo",temperature=0): # Andrew mentioned that the prompt/ completion paradigm is preferable for this class
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

def translator_ed(english_text):
    prompt = f"""
                You are a translator that specializes in Moroccan Darija, created to provide precise translations from English to dialect of Arabic spoken in Morocco.\

                Translate the sentence between ''.\

                sentence = '{english_text}'
                        """
    return get_completion(prompt)


def translator_de(darija_text):
    prompt = f"""
                You are a translator that specializes in Moroccan Darija, created to provide precise translations from dialect of Arabic spoken in Morocco to English.\

                Translate the sentence between ''.\

                sentence = '{darija_text}'
                        """
    return get_completion(prompt)


def DarijaGPT(darija_text):
    
    question = translator_de(darija_text)
    
    prompt = f"""
            You are a wikibot, created to provide short answers to any question.\
            
            Anser the question between ''.\
            
            You shouldn't write more than 20 words.\
            
            question = '{question}'
        """
    answer = get_completion(prompt)
    
    return translator_ed(answer)


print(DarijaGPT("Wahd Salaaaamo 3alikom"))

