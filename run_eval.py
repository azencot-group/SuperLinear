#!/usr/bin/env python
# -*- coding:utf-8 -*-
import argparse
import logging
import os

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from run_files.datasets.benchmark_dataset import BenchmarkEvalDataset, GeneralEvalDataset


def count_num_tensor_elements(tensor):
    return tensor.numel()


# ------------------ Metrics ------------------
class SumEvalMetric:
    def __init__(self, name, init_val: float = 0.0):
        self.name = name
        self.value = init_val

    def push(self, preds, labels, **kwargs):
        self.value += self._calculate(preds, labels, **kwargs)

    def _calculate(self, preds, labels, **kwargs):
        pass


class MSEMetric(SumEvalMetric):
    def _calculate(self, preds, labels, **kwargs):
        return torch.sum((preds - labels) ** 2)


class MAEMetric(SumEvalMetric):
    def _calculate(self, preds, labels, **kwargs):
        return torch.sum(torch.abs(preds - labels))


class SuperLinear:
    def __init__(self, model_path, device, context_length, prediction_length, **kwargs):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype='auto',
            trust_remote_code=True,
            force_download=True
        )

        self.model = model
        self.model.backbone.inf_pred_len = prediction_length
        self.device = device
        self.prediction_length = prediction_length
        self.model.eval()

    def predict(self, batch):
        series = batch['inputs'].to(self.device).to(self.model.dtype)
        labels = batch['labels'].to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs_embeds=series)
        preds = outputs.logits

        if preds.dim() > labels.dim():
            labels = labels[..., None]
        return preds, labels


def evaluate(args):
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Metrics
    metric_list = [
        MSEMetric(name='mse'),
        MAEMetric(name='mae'),
    ]

    # Model
    model = SuperLinear(
        args.model,
        device=device,
        context_length=args.context_length,
        prediction_length=args.prediction_length
    )

    # Dataset
    if args.data.endswith('.csv'):
        dataset = BenchmarkEvalDataset(
            args.data,
            context_length=args.context_length,
            prediction_length=args.prediction_length,
        )
    else:
        dataset = GeneralEvalDataset(
            args.data,
            context_length=args.context_length,
            prediction_length=args.prediction_length,
        )

    test_dl = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        prefetch_factor=2,
        drop_last=False,
    )

    acc_count = 0
    for batch in tqdm(test_dl, desc="Evaluating"):
        preds, labels = model.predict(batch)
        for metric in metric_list:
            metric.push(preds, labels)
        acc_count += count_num_tensor_elements(preds)

    # Final metrics
    final_results = {metric.name: (metric.value / acc_count).item() for metric in metric_list}

    print(f"Evaluation results:\n{final_results}")

    # Optionally log
    logging.info({
        'model': args.model,
        'data': args.data,
        'context_length': args.context_length,
        'prediction_length': args.prediction_length,
        **final_results,
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SuperLinear Evaluate')
    parser.add_argument('--model', '-m', type=str, default='SequentialLearning/SuperLinear', help='Model path')
    parser.add_argument('--data', '-d', type=str, required=True, help='Benchmark data path')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='Batch size of evaluation')
    parser.add_argument('--context_length', '-c', type=int, default=512, help='Context length')
    parser.add_argument('--prediction_length', '-p', type=int, default=96, help='Prediction length')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    evaluate(args)
