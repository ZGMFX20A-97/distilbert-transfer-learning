import marimo

__generated_with = "0.17.7"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 必要なライブラリをインポート
    """)
    return


@app.cell
def _():
    import huggingface_hub
    import datasets,evaluate,accelerate
    import gradio as gr
    import random
    import numpy as np
    import pandas as pd
    import torch
    import transformers
    return evaluate, huggingface_hub, np, pd


@app.cell
def _(huggingface_hub):
    # huggingfaceへログインする
    # huggingfaceで自分自身のトークンを作成する必要がある
    # このセルを実行したらログイン画面が出る

    huggingface_hub.login()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### データセットの取得
    """)
    return


@app.cell
def _():
    from datasets import load_dataset

    dataset = load_dataset("mrdbourke/learn_hf_food_not_food_image_captions")
    dataset
    return (dataset,)


@app.cell
def _(dataset, pd):
    df = pd.DataFrame(dataset["train"])
    df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### データの前処理
    """)
    return


@app.cell
def _(dataset):
    # ラベルを数値にマッピングする

    id2label = {idx:label for idx,label in enumerate(dataset["train"].unique("label")[::-1])}
    label2id = {label:idx for idx,label in id2label.items()}
    print(id2label)
    print(label2id)
    return id2label, label2id


@app.cell
def _(label2id):
    # ラベルを数値に変換する関数を定義する

    def map_labels_to_number(example):
      example["label"] = label2id[example["label"]]
      return example
    return (map_labels_to_number,)


@app.cell
def _(dataset, map_labels_to_number):
    dataset_1 = dataset['train'].map(map_labels_to_number)
    dataset_1[:5]
    return (dataset_1,)


@app.cell
def _(dataset_1):
    # 訓練データ、テストデータに分割する
    dataset_2 = dataset_1.train_test_split(test_size=0.2, seed=42)
    dataset_2
    return (dataset_2,)


@app.cell
def _():
    # 形態素解析(tokenizer)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="distilbert/distilbert-base-uncased",use_fast=True)
    tokenizer
    return (tokenizer,)


@app.cell
def _(tokenizer):
    tokenizer("I love cat!")
    return


@app.cell
def _(tokenizer):
    # tokenizerのボキャブラリの長さ
    length_of_tokenizer_vocab = len(tokenizer.vocab)
    print(length_of_tokenizer_vocab)

    # tokenizerが捌けるシーケンスデータの最大長さ
    max_tokenizer_input_sequence_length = tokenizer.model_max_length
    print(max_tokenizer_input_sequence_length)
    return


@app.cell
def _(tokenizer):
    # tokenizeの関数を定義
    def tokenize_text(examples):
      return tokenizer(examples["text"],
                       padding=True,# padding: 短いテキストの長さを上限に揃える
                       truncation=True) # truncation: max_tokenizer_input_sequence_lengthに達したテキストは切り取る
    return (tokenize_text,)


@app.cell
def _(dataset_2, tokenize_text):
    tokenized_dataset = dataset_2.map(tokenize_text, batched=True, batch_size=250)
    tokenized_dataset
    return (tokenized_dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 評価指標の定義
    """)
    return


@app.cell
def _(evaluate, np):
    from typing import Tuple
    accuracy = evaluate.load('accuracy')

    # Accuracy
    def compute_accuracy(predictions_and_labels: Tuple[np.array, np.array]):
        pred, label = predictions_and_labels
        if len(pred.shape) >= 2:
            pred = np.argmax(pred, axis=1)
        return accuracy.compute(predictions=pred, references=label)
    return (compute_accuracy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### モデルのセットアップ
    """)
    return


@app.cell
def _(id2label, label2id):
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path="distilbert/distilbert-base-uncased",
        num_labels=2,
        id2label=id2label,
        label2id=label2id
    )
    return (model,)


@app.cell
def _(model):
    # モデルの構成
    # embedding - 単語や文といった自然言語の情報を、その単語や文の意味を表現するベクトル空間に配置する
    # transformer - モデルアーキテクチャの中核、embedding内の相関関係を発見する
    # classifier - このレイヤーで自分のタスク適用させる

    model
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### モデル内のパラメーター数
    """)
    return


@app.cell
def _(model):
    def count_params(model):
      trainable_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
      total_parameters = sum(param.numel() for param in model.parameters())

      return {"trainable_parameters": trainable_parameters,"total_parameters": total_parameters}

    count_params(model)

    # すべてのパラメータは学習可能
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### モデルを保存するためのフォルダ
    """)
    return


@app.cell
def _():
    from pathlib import Path

    # モデルのディレクトリを作成
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # モデル名
    model_save_name = "learn_hf_food_not_food_text_classifier"

    # モデルを保存するパス
    model_save_dir = Path(models_dir,model_save_name)

    model_save_dir
    return (model_save_dir,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ハイパーパラメータの設定
    """)
    return


@app.cell
def _(model_save_dir):
    from transformers import TrainingArguments

    print(f"[INFO] Saving model checkpoint: {model_save_dir}")

    BATCH_SIZE = 32

    training_args = TrainingArguments(
        output_dir=model_save_dir,
        learning_rate=1e-4,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        use_cpu=False,
        seed=42,
        load_best_model_at_end=True,
        logging_strategy="epoch",
        report_to="none",
        hub_private_repo=False
    )
    return (training_args,)


@app.cell
def _(training_args):
    training_args
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Trainerインスタンスの作成
    """)
    return


@app.cell
def _(compute_accuracy, model, tokenized_dataset, tokenizer, training_args):
    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_accuracy
    )

    trainer
    return (trainer,)


@app.cell
def _(trainer):
    trainer.train()
    return


@app.cell
def _(tokenized_dataset, trainer):
    predictions_all = trainer.predict(tokenized_dataset["test"])
    predictions_values = predictions_all.predictions
    prediction_metrics = predictions_all.metrics

    print(f"[INFO] Prediction metrics on the test data:")
    prediction_metrics
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
