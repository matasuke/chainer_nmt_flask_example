import argparse
import json

import numpy
from flask import Flask, jsonify, request, render_template
from janome.tokenizer import Tokenizer

from inference import sequence_embed, load_vocabulary
from inference import Seq2seq

import chainer
from chainer import serializers

app = Flask(__name__)

def _predict(source):
    with chainer.using_config('train', False):
        utterance = [ token.surface for token in ja_tokenizer.tokenize(source) ]

        tokens = [inv_source_words[u] if u in inv_source_words else inv_source_words['<UNK>'] for u in utterance]
        tokens = numpy.expand_dims(numpy.array(tokens), axis=0)

        result = model.inference(tokens)[0]
        result_sentence = ' '.join([target_words[y] for y in result])

    return result_sentence

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if request.form['title']:
            title = request.form['title']
            result = _predict(title)

            return render_template('index.html', source=title, target=result)
        else:
            return render_template('index.html')
    else:
        return render_template('index.html')

@app.route('/title2history', methods=['POST'])
def title2history():
    data = json.loads(request.data)
    source = data["title"]

    result = _predict(source)
    result = {
      "Content-Type": "application/json",
      "History": result
    }

    return jsonify(result)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Chainer example: seq2seq')
    parser.add_argument('SOURCE_VOCAB', help='source vocabulary file')
    parser.add_argument('TARGET_VOCAB', help='target vocabulary file')
    parser.add_argument('SNAPSHOT', help='snapshot path')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--unit', '-u', type=int, default=512,
                        help='number of units')
    parser.add_argument('--layer', '-l', type=int, default=1,
                        help='number of layers')
    args= parser.parse_args()
    source_ids = load_vocabulary(args.SOURCE_VOCAB)
    target_ids = load_vocabulary(args.TARGET_VOCAB)

    target_words = {i: w for w, i in target_ids.items()}
    source_words = {i: w for w, i in source_ids.items()}

    inv_source_words = {w: i for w, i in source_ids.items()}
    inv_target_words = {w: i for w, i in target_ids.items()}

    # Setup model
    model = Seq2seq(args.layer, len(source_ids), len(target_ids), args.unit)
    if args.gpu >= 0:
        chainer.backends.cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)

    serializers.load_npz(args.SNAPSHOT, model, path='updater/model:main/')
    ja_tokenizer = Tokenizer()

    app.run(host='0.0.0.0',port=5000,debug=True)
