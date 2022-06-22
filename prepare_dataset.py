import os
import json
import yaml
import nltk
import sentencepiece as spm
from datasets import load_dataset
from collections import defaultdict




def save_json(data, f_name):
    with open(f'data/{f_name}', 'w') as f:
        json.dump(data, f)




def build_vocab(concat_data):
    with open('data/concat.txt', 'w') as f:
        f.write('\n'.join(concat_data))

    with open("vocab.yaml", 'r') as f:
        vocab_configs=yaml.safe_load(f)

    opt = f"--input=data/concat.txt \
            --model_prefix=data/spm \
            --vocab_size={vocab_configs['vocab_size']} \
            --character_coverage={vocab_configs['coverage']} \
            --model_type=bpe \
            --unk_id={vocab_configs['unk_id']} --unk_piece={vocab_configs['unk_piece']} \
            --pad_id={vocab_configs['pad_id']} --pad_piece={vocab_configs['pad_piece']} \
            --bos_id={vocab_configs['bos_id']} --bos_piece={vocab_configs['bos_piece']} \
            --eos_id={vocab_configs['eos_id']} --eos_piece={vocab_configs['eos_piece']}"

    spm.SentencePieceTrainer.Train(opt)
    os.remove('data/concat.txt')




def preprocess(orig_data):
    train, valid, test = orig_data['train'], orig_data['validation'], orig_data['test']

    train_src, train_trg = train['article'][::2][:100000], train['highlights'][::2][:100000]
    valid_src, valid_trg = valid['article'][::3][:3000], valid['highlights'][::3][:3000]
    test_src, test_trg = test['article'][::3][:3000], test['highlights'][::3][:3000]

    src = train_src + valid_src + test_src
    trg = train_trg + valid_trg + test_trg
    concat = src + trg

    build_vocab(concat)
    
    return src, trg




def process(src, trg, tokenizer):
    nltk.download('punkt')
    
    data = []
    for orig, summ in zip(src, trg):
        elem = defaultdict(list)
        
        elem['orig_src'], elem['orig_trg'] = [orig], [summ]
        
        split_src, split_trg = nltk.tokenize.sent_tokenize(orig), nltk.tokenize.sent_tokenize(summ)
        split_src_ids = tokenizer.Encode(split_src)
        split_trg_ids = tokenizer.Encode(split_trg)

        for seq in split_src_ids:
            elem['src'].extend(seq)

        for seq in split_trg_ids:
            elem['trg'].extend(seq)
        
        data.append(elem)

    train, valid, test = data[:-6000], data[-6000:-3000], data[-3000:]
    
    save_json(train, 'train.json')
    save_json(valid, 'valid.json')
    save_json(test, 'test.json')





def main():
    orig_data = load_dataset('cnn_dailymail', '3.0.0')
    data = preprocess(orig_data)
    
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load('data/spm.model')
    tokenizer.SetEncodeExtraOptions('bos:eos')
    
    process(data, tokenizer)
    



if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    main()