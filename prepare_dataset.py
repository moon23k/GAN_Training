import os
import json
import nltk
from datasets import load_dataset
from collections import defaultdict




def save_json(data, f_name):
    os.makedirs('data', exist_ok=True)
    with open(f'data/{f_name}', 'w') as f:
        json.dump(data, f)



def build_vocab(concat_data):
    with open('data/concat.txt', 'w') as f:
        f.write('\n'.join(concat_data))

    with open("vocab.yaml", 'r') as f:
        vocab_configs=yaml.safe_load(f)

    opt = f"--input=data/concat.txt \
            --model_prefix=spm \
            --vocab_size={vocab_configs.vocab_size} \
            --character_coverage={vocab_configs.coverage} \
            --model_type=bpe \
            --unk_id={vocab_configs.unk_id} --unk_piece={vocab_configs.unk_piece} \
            --pad_id={vocab_configs.pad_id} --pad_piece={vocab_configs.pad_piece} \
            --bos_id={vocab_configs.bos_id} --bos_piece={vocab_configs.bos_piece} \
            --eos_id={vocab_configs.eos_id} --eos_piece={vocab_configs.eos_piece}"

    spm.SentencePieceTrainer.Train(opt)
    os.remove('data/concat.txt')



def preprocess(orig_data):
    nltk.download('punkt')
    train, valid, test = orig_data['train'], orig_data['validation'], orig_data['test']
    
    src, trg = [], []
    data, concat = [], []

    train_src, train_trg = train['article'], train['highlights']
    valid_src, valid_trg = valid['article'], valid['highlights']
    test_src, test_trg = test['article'], test['highlights']

    src.extend(train_src + valid_src + test_src)
    trg.extend(train_trg + valid_trg + test_src)

    for orig, summ in zip(src, trg):
        elem = defaultdict(list)
        
        orig_seq = nltk.tokenize.sent_tokenize(orig)
        summ_seq = nltk.tokenize.sent_tokenize(summ)
        
        elem['orig_src'].extend(orig_seq)
        elem['orig_trg'].extend(summ_seq)
        
        data.append(elem)
        concat.extend(orig_seq + summ_seq)

    build_vocab(concat)
    return data



def process(data, tokenizer):
    for elem in data:
        
        src_ids, trg_ids = [], []
        for seq in elem['orig_src']:
            src_ids += tokenizer.Encode(seq)
        for seq in elem['orig_trg']:
            trg_ids += tokenizer.Encode(seq)
        
        elem['src'].extend(src_ids)
        elem['src'].extend(trg_ids)

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
    main()