import math, torch, evaluate
from module import load_dataloader



class Tester:
    def __init__(self, config, model, tokenizer):
        super(Tester, self).__init__()
        
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = load_dataloader(
            config, tokenizer, 'test', shuffle=False
        )

        self.device = config.device
        self.pad_id = config.pad_id
        self.bos_id = config.bos_id
        self.eos_id = config.eos_id
        self.max_len = config.max_len
        
        self.metric_module = evaluate.load('bleu')
        

    def test(self):
        score = 0.0         
        self.model.eval()

        with torch.no_grad():
            for batch in self.dataloader:
                pred = self.predict(batch['x'].to(self.device))
                score += self.evaluate(pred, batch['y'])

        txt = f"TEST Result\n"
        txt += f"-- Score: {round(score/len(self.dataloader), 2)}\n"
        print(txt)


    def tokenize(self, batch):
        return [self.tokenizer.decode(x) for x in batch.tolist()]


    def evaluate(self, pred, label):

        pred = self.tokenize(pred)
        if pred == ['' for _ in range(len(pred))]:
            return 0.0

        label = self.tokenize(label)

        score = self.metric_module.compute(
            predictions=pred, 
            references =[[l] for l in label]
        )['bleu']

        return score * 100


    def predict(self, x):

        batch_size = x.size(0)
        pred = torch.zeros((batch_size, self.max_len)).fill_(self.pad_id)
        pred = pred.type(torch.LongTensor).to(self.device)
        pred[:, 0] = self.bos_id

        e_mask = self.model.pad_mask(x)
        memory = self.model.encode(x, e_mask)

        for idx in range(1, self.max_len):
            y = pred[:, :idx]
            d_out, _ = self.model.decode(y, memory, None, e_mask, use_cache=False)

            logit = self.model.generator(d_out)
            pred[:, idx] = logit.argmax(dim=-1)[:, -1]

            if (pred == self.eos_id).sum().item() == batch_size:
                break

        return pred        
