import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .trainer import TrainerBase
from module import load_generator



class DisTrainer(TrainerBase):
	def __init__(self, config, model, train_dataloader, valid_dataloader):
		super(DisTrainer, self).__init__(config)
		
		#model load시, initialized 된 모델이랑 학습된 모델이랑 선택사항을 옵션으로 주는게 나을 수도
		self.generator = load_generator(config) #load pretrianed generator
		self.generator.eval()

		self.model = model

		self.train_dataloader = train_dataloader
		self.valid_dataloader = valid_dataloader

		self.optimizer
		self.lr_scheduler

		self.ckpt



	def collate(self, batch):
		sample_input = batch[0].to(self.device)
		label = batch[1].to(self.device)

		#label을 주는 이유는 max_len을 어느정도 유사하게 생성하도록 유도하기 위함
		sample = self.generator.generate(sample_input, label) 


		#shuffling
		random_indice = torch.random()

		x = torch.cat(label, sample)[random_indice]
		y = torch.zeros()

		return x, y


	def train_epoch(self):
		self.model.train()

		for idx, batch in enumerate(self.train_dataloader):
			x, y = self.collate(batch)
			loss = self.model(x, y).loss


		return


	def valid_epoch(self):
		self.model.eval()

		with torch.no_grad():
			for batch in self.valid_dataloader:
				x, y = self.collate(batch)
				loss = self.model(x, y)

		return		