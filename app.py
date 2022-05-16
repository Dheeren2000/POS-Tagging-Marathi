from flask import Flask, render_template, flash, request ,make_response
from nltk.tag import tnt
from nltk.corpus import indian
import nltk
import pandas as pd
import numpy as np

nltk.download('indian')
nltk.download('punkt')


 
# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
 
def marathi_model():
    train_data = indian.tagged_sents('marathi.pos')
    tnt_pos_tagger = tnt.TnT()
    tnt_pos_tagger.train(train_data)
    return tnt_pos_tagger
   
model = marathi_model()
    
@app.route("/", methods=['GET', 'POST'])
def hello():
    return render_template('index.html')
 
@app.route('/tag',methods = ['POST', 'GET'])
def result():
    if request.method == 'POST':
        result = request.form
        sentence=result['sentence']

        global taggedOutput
        taggedOutput=array=np.array((model.tag(nltk.word_tokenize(sentence))))
        pd.DataFrame(taggedOutput).to_csv("file.csv")
        print("Sentence : ", sentence)
        print("Tagged Output", taggedOutput)
        print(nltk.word_tokenize(sentence))
        output={
            "input":sentence,
            "taggedOutput":taggedOutput
        }
        return render_template("result.html",output=output )


if __name__ == "__main__":
    app.run()
