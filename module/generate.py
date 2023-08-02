import json, torch



def generate(config, generator, tokenizer, split):

    #Load Pretrained Generator Model States
    model_state = torch.torch.load(config.g_base_ckpt, map_location=config.device)['model_state_dict']
    model.load_state_dict(model_state)
    model.eval()

    config.mode = 'generate'
    dataloader = load_dataloader(config)

    generated = []
    for batch in dataloader:
        uttr, resp = batch[0], batch[1]
        batch_size = len(uttr)

        uttr_encodings = tokenizer(uttr, padding=True, truncation=True, 
                                   return_tensors='pt').to(config.device)        
        
        with torch.no_grad():
            pred = model.generate(
                input_ids=uttr_encodings.input_ids,
                attention_mask=uttr_encodings.attention_mask,
                use_cache=True
            )

        pred = tokenizer.batch_decode(pred, skip_special_tokens=True)

        for i in range(batch_size):
            generated.append({'src': uttr[i], 
                              'trg': resp[i], 
                              'pred': pred[i]})

    config.mode = 'pretrain'
    with open(f'data/dis_{split}.json', 'w') as f:
        json.dump(generated, f)