import math, torch, evaluate



class Tester:
    def __init__(self, config, model, tokenizer, test_dataloader):
        super(Tester, self).__init__()
        
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = test_dataloader

        self.device = config.device
        self.max_len = config.max_len
        
        self.metric_name = 'BLEU'
        self.metric_module = evaluate.load('bleu')
        

    def test(self):
        score = 0.0         
        self.model.eval()

        with torch.no_grad():
            for batch in self.dataloader:
                pred = self.model.generate(batch['src'].to(self.device))
                score += self.evaluate(pred, batch['trg'])

        txt = f"TEST Result\n"
        txt += f"-- Score: {round(score/len(self.dataloader), 2)}\n"
        print(txt)


    def tokenize(self, batch):
        return [self.tokenizer.decode(x) for x in batch.tolist()]


    def evaluate(self, pred, label):
        pred = self.tokenize(pred)
        label = self.tokenize(label)

        score = self.metric_module.compute(
            predictions=pred, 
            references =[[l] for l in label]
        )['bleu']

        return score * 100