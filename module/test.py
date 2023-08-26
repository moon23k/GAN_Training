import torch, evaluate



class Tester:
    def __init__(self, config, g_model, d_model, tokenizer, test_dataloader):

        self.device = config.device
        self.max_len = config.max_len

        self.g_model = g_model
        self.d_model = d_model
        
        self.tokenizer = tokenizer
        self.dataloader = test_dataloader





    def tokenize(self, tokenizer, tokenizer_inputs):
        return tokenizer(
            tokenizer_inputs, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        ).to(self.device)


    def test(self):
        scores = 0

        self.g_model.eval()
        self.d_model.eval()

        with torch.no_grad():
            for idx, batch in enumerate(self.dataloader):   
                uttr, resp = batch[0], batch[1]

                #tokenize inputs for generator
                g_uttr_encodings = self.tokenizer(self.tokenizer, uttr)
                g_ids = g_uttr_encodings.input_ids
                g_masks = g_uttr_encodings.attention_mask

                #generate predictions
                preds = self.g_model.generate(
                    input_ids=g_ids,
                    attention_mask=g_masks, 
                    max_new_tokens=self.max_len, 
                    use_cache=True
                )

                #Decode generator predictions
                preds = self.g_tokenizer.batch_decode(preds, skip_special_tokens=True)

                #Tokenize inputs for discriminator
                d_encodings = self.tokenize(self.tokenizer, preds)
                d_ids = d_encodings.input_ids
                d_masks = d_encodings.attention_mask
                logits = self.d_model(input_ids=d_ids, attention_mask=d_masks)
                scores += logits[logits > 0.5].sum()

        scores = scores / len(dataloader)

        print('Test Results')
        print(f"  >> Test Score: {scores:.2f}")
