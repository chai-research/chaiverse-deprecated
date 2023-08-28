[![Guanaco Banner](https://imgur.com/wJHIeAU.png)](https://www.chai-research.com/competition.html)
[![Pull Requests Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](http://makeapullrequest.com)
[![first-timers-only Friendly](https://img.shields.io/badge/first--timers--only-friendly-blue.svg)](http://www.firsttimersonly.com/)

[ChaiVerse](https://www.chai-research.com/competition.html) is the community's one-stop repository for large language model training, deployment and REAL LIVE USER evaluation package. Train models and win prizes!


## Quick Start

To train a LLaMA7b model and push it to huggingface it's just 5 lines of code 🥳

```python
import chaiverse as cv

dataset = cv.load_dataset('ChaiML/davinci_150_examples', 'chatml')

model = cv.LLaMA7b()
model.fit(dataset, 'dummy_run', num_epochs=1)

model_url = 'ChaiML/llama7b_dummy'
model.push_to_hub(model_url, private=True)
```
