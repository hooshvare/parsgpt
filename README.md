<h1 align="center">ParsGPT 🦁</h1>

<br/><br/>


# How to use
You can use this model directly with a pipeline for text generation:

## Installing requirements

```bash
pip install -U transformers
pip install -U hazm
```

## How to generate using pipeline

```python
from transformers import pipeline
import hazm


normalizer = hazm.Normalizer(persian_numbers=False)

def normalize_input(text):
    text = normalizer.normalize(text)
    return text

def sents_as_output(text, num_sents=1):
    sents = hazm.sent_tokenize(text)
    if num_sents > 0:
        return " ".join(sents[:num_sents])
    return " ".join(sents[0])


generator = pipeline('text-generation', "HooshvareLab/gpt2-fa")
text = "در یک اتفاق شگفت انگیز، پژوهشگران"
text = normalize_input(text)

outputs = generator(text)
for output in outputs:
    generated = output["generated_text"]
    print(sents_as_output(generated))
```

```text
در یک اتفاق شگفت انگیز، پژوهشگران قصد دارند با استفاده از داده‌های حاصل از چندین تلسکوپ، عکس‌هایی با وضوح مختلف از سیاره‌ی مشتری و زحل تهیه کنند.
```

# GPT2 Visualization

This visualization is powered by [Ecco](https://github.com/jalammar/ecco) (an interactive language modeling visualization).

|      Notebook      |                                                                                                                                                                                                 |
|:------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| Ecco Visualization | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hooshvare/parsgpt/blob/master/notebooks/Persian_GPT2_Visualization.ipynb) |
# How to fine-tune

| Notebook       |                                                                                                                                                                                                |
|----------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Simple Version | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hooshvare/parsgpt/blob/master/notebooks/Persian_Poetry_FineTuning.ipynb) |

## Examples - Models
If you fine-tuned gpt2-fa on your dataset, share it with us. All things you need to do, create a pull request and share yours with us. We are looking forward to it.

|                                        Model                                        |                                                                                                           Description                                                                                                           |                                                                                        How to Use                                                                                        |
|:-----------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|  [HooshvareLab/gpt2-fa-poetry](https://huggingface.co/HooshvareLab/gpt2-fa-poetry)  |                                                 The model was fine-tuned on [ChronologicalPersianPoetryDataset](https://github.com/aghasemi/ChronologicalPersianPoetryDataset).                                                 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hooshvare/parsgpt/blob/master/notebooks/Persian_Poetry_GPT2.ipynb) |
| [HooshvareLab/gpt2-fa-comment](https://huggingface.co/HooshvareLab/gpt2-fa-comment) | The model can generate comments based on your aspects, and the model was fine-tuned on [persiannlp/parsinlu]( https://github.com/persiannlp/parsinlu ). Currently, the model only supports aspects in the food and movie scope. |                                                                                                                                                                                          |

# Citation

Please cite in publications as the following:

```bibtex
@misc{ParsGPT2,
  author = {Hooshvare Team},
  title = {ParsGPT2, a Persian version of GPT2},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/hooshvare/parsgpt}},
}
```

# Questions?
Post a Github issue on the [ParsGPT2 Issues](https://github.com/hooshvare/parsgpt/issues) repo.
