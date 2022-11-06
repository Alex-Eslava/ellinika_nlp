from email.utils import parsedate_to_datetime
import os
import json
import time
import logging
import random
import re
import yaml
import argparse

import numpy as np
import pandas as pd
import pytorch_lightning as pl # 0.8.1
import evaluate

from numpy.random import RandomState
from torch.utils.data import Dataset, Dataloader
from transformers import (
    AdamW,
    MT5forConditionalGeneration,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    pipeline
)

class GenerativeDataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, max_len=30):
        self.path = os.path.join(data_dir, type_path = '.csv')
        
        # Hardcoded for QGen, TODO: rework to generic
        self.context = 'context'
        self.target = 'question'
        if type_path =='.csv':
            self.data = pd.read_csv(self.path)
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

    def __len__(self):
        return len(self.inputs)

    
    def __getitem__(self, index: Any):
        source_ids = self.inputs[index]['input_ids'].squeeze()
        target_ids = self.targets[index]['input_ids'].squeeze()
        src_mask = self.inputs[index]['attention_mask'].squeeze()
        target_mask = self.targets[index]['attention_mask'].squeeze()
        return {'source_ids': source_ids, 'source_mask': src_mask, 'target_ids': target_ids, 'target_mask': target_mask}

    def _build(self):
        for idx in range(len(self.data)):
            input_text, output_text = self.data.loc[idx, self.context], self.data.loc[idx, self.target]

            _input = input_text
            target = output_text
            # TODO: Parametrize max_length for both inpt + target
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [_input], max_length = 200, pad_to_max_length=True, return_tensors='pt'
            )
            tokenized_targets = self.tokenizer.batch_encode_plus(
                [target], max_length = 20, pad_to_max_length=True, return_tensors = 'pt'
            )

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)

class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        self.hparams = hparams
        self.model = MT5forConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(hparams.tokenizer_name_or_path)

    def is_logger(self):
        return True
    
    def forward(
        self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )
    
    def _step(self, batch):
        labels = batch['target_ids']
        # necessary hack 
        labels[labels[:,:] == self.tokenizer.pad_token_id] = -100
        outputs = self(
            input_ids=batch['source_ids'],
            attention_mask=batch['source_mask'],
            labels=labels,
            decoder_attention_mask=batch['target_mask']
        )
        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        tensorboard_logs = {'train_loss':loss}
        return {'loss':loss, 'log':tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_train_loss':avg_train_loss}
        return {'avg_train_loss': avg_train_loss, 'log':tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {'val_loss':loss}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss':avg_loss}
        return {'avg_val_loss':avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        model = self.model
        no_decay = ['bias','LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.hparams.weight_decay,
            },
            {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            }
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx,optimizer, optimizer_idx, second_order_closure=None):
        # TODO: Improve considering Accelerate + TPU implementations 
        optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {
            'loss': "{:.3f}".format(self.trainer.avg_loss),
            'lr': self.lr_scheduler.get_last_lr()[-1]
        }
        return tqdm_dict

    def train_dataloader(self):
        train_dataset = get_dataset(tokenizer=self.tokenizer, type_path='train', args=self.hparams)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, 
                                drop_last=True, shuffle=True, num_workers=4)
        t_total = (
            (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
            // self.hparams.gradient_accumulation_steps * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader
    
    def val_dataloader(self):
        val_dataset = get_dataset(tokenizer=self.tokenizer, type_path='valid', args=self.hparams)
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)

    logger = logging.getLogger(__name__)

    class LoggingCallback(pl.Callback):
        def on_validation_end(self, trainer, pl_module):
            logger.info("---- Validation Results ----")
            if pl_module.is_logger():
                metrics = trainer.callback_metrics
                for key in sorted(metrics):
                    if key not in ['log']:
                        logger.info("{} = {}\n".format(key, str(metrics[key])))

        def on_test_end(self, trainer, pl_module): 
            logger.info("---- Test Results ----")
            if pl_module.is_logger():
                metrics = trainer.callback_metrics
                output_test_results_filepath = os.path.join(pl_module.hparams.output_dir, 'test_results.txt')
                with open(output_test_results_filepath, "w") as writer:
                    for key in sorted(metrics):
                        if key not in ['log']:
                            logger.info("{} = {}\n".format(key, str(metrics[key])))
                            writer.write("{} = {}\n".format(key, str(metrics[key])))

    def get_dataset(tokenizer, type_path, args):
        return GenerativeDataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path, max_len=args.max_seq_length)

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def generate_output(inp_ids, attn_mask, model):
        """
        Props to this guy--> https huggingface co/blog/how-to-generate
        """
        output = model.generate(input_ids = inp_ids, 
                                attention_mask=attn_mask,
                                do_sample=True,
                                max_length=50,
                                top_p = 0.93,
                                top_k = 50,
                                num_return_sequences = 5,
                                min_length = 3,
                                temperature = 0.9,
                                repetition_penalty = 1.2,
                                length_penalty = 1.5,
                                no_repeat_ngram_size = 2,
                                num_beams = 4
        )
        decoded_output = [tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True) for out in output] 
        return [clean_output.strip() for clean_output in decoded_output]

    def t5_generate(input_text, model, tokenizer):
        encoding = tokenizer.encode_plus(input_text, return_tensors='pt')
        input_ids, attention_masks = encoding['input_ids'].to(device), encoding['attention_mask'].to(device)
        output = generate_output(input_ids, attention_masks, model)
        return output

    if __name__ == "__main__":

        config_path = 'qgen_config.yaml'
        config = yaml.safe_load(open(config_path))
        set_seed(config['seed'])
        dataset_path = config['dataset_path']
        interim_data_path = config['interim_path']
        result_save_dir_path = config['output_path']

        args_dict = dict(
            data_dir = interim_data_path,
            output_dir = result_save_dir_path,
            model_name_or_path = config['model_ckpt'],
            tokenizer_name_or_path = config['model_ckpt'],
            max_seq_length = config['train_args']['max_seq_length'],
            learning_rate = config['train_args']['learning_rate'],
            weight_decay = config['train_args']['weight_decay'],
            adam_epsilon = config['train_args']['adam_epsilon'],
            warmup_steps = config['train_args']['warmup_steps'],
            train_batch_size = config['train_args']['train_batch_size'],
            eval_batch_size = config['train_args']['eval_batch_size'],
            num_training_epochs = config['train_args']['num_training_epochs'],
            gradient_accumulation_steps = config['train_args']['gradient_accumulation_steps'],
            n_gpu=config['n_gpus'],
            early_stop_callback=False,
            fp_16 = False, 
            opt_level=config['train_args']['opt_level'],
            max_grad_norm = 1.0,
            seed=config['seed']
        )

        # Couldn't be bothered to import sklearn U_U
        data = pd.read_csv(dataset_path, sep='|')
        train = data.sample(frac=0.8, random_state=RandomState())
        test = data.loc[~data.index.isin(train.index)]

        train_path = f"{interim_data_path}/train.csv"
        test_path = f"{interim_data_path}/test.csv"
        train.to_csv(train_path)
        test.to_csv(test_path)

        args = argparse.Namespace(**args_dict)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            period=1, filepath=args.output_dir, prefix='checkpoint', monitor='val_loss', mode='min', save_top_k=1
        )
        train_params = dict(
            accumulate_grad_batches = args.gradient_accumulation_steps,
            gpus = config['n_gpus'],
            max_epochs = args.num_train_epochs,
            early_stop_callback=False,
            precision = 16 if args.fp_16 else 32,
            amp_level=args.opt_level,
            gradient_clip_val=args.max_grad_norm,
            checkpoint_callback=checkpoint_callback,
            callbacks=[LoggingCallback()]
        )
        # Training yay
        model = T5FinetUNER(args)
        trainer = pl.Trainer(**train_params)
        trainer.fit(model)
        model.model.save_pretrained(result_save_dir_path)
        model.tokenizer.save_pretrained(result_save_dir_path)
        # TODO: Add BLEU metric


