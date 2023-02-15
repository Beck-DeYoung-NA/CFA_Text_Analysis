import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
import pytorch_lightning as pl

class MultiLabelDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer: BertTokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        label = self.labels[index]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return dict(
            input_ids=encoding['input_ids'].flatten(),
            attention_mask=encoding['attention_mask'].flatten(),
            labels=torch.tensor(label, dtype=torch.float)
        )

class MultiLabelBert(pl.LightningModule):
    def __init__(self, num_labels=3, lr=2e-5):
        super().__init__()
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
        self.lr = lr

    def forward(self, input_ids, attention_mask):
        return self.bert(input_ids, attention_mask=attention_mask)

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = self(input_ids, attention_mask)
        loss = torch.nn.BCEWithLogitsLoss()(outputs.logits, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = self(input_ids, attention_mask)
        loss = torch.nn.BCEWithLogitsLoss()(outputs.logits, labels)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = self(input_ids, attention_mask)
        predicted_labels = torch.round(torch.sigmoid(outputs.logits))
        accuracy = (predicted_labels == labels).float().mean()
        self.log('test_acc', accuracy)
        return accuracy

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

# Prepare data
texts = ['text1', 'text2', 'text3', ...]
labels = [[1, 0, 1], [0, 1, 1], [1, 1, 0], ...]
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = MultiLabelDataset(texts, labels, tokenizer, max_len=256)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4)