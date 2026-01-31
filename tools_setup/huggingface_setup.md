# HuggingFace Setup Guide

## What is HuggingFace?

HuggingFace is the leading platform for:
- Pre-trained models (100,000+ models)
- Datasets (10,000+ datasets)
- Transformers library
- Model hosting and deployment

## Installation

```bash
pip install transformers datasets tokenizers accelerate
```

## Quick Start

### 1. Text Generation

```python
from transformers import pipeline

# Load model
generator = pipeline('text-generation', model='gpt2')

# Generate text
output = generator("Artificial intelligence is", max_length=50)
print(output[0]['generated_text'])
```

### 2. Text Classification

```python
classifier = pipeline('sentiment-analysis')

result = classifier("I love this product!")
print(result) # [{'label': 'POSITIVE', 'score': 0.9998}]
```

### 3. Image Generation

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
 "runwayml/stable-diffusion-v1-5",
 torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

image = pipe("A beautiful sunset").images[0]
image.save("sunset.png")
```

## HuggingFace Hub

### Browse Models

Visit: [huggingface.co/models](https://huggingface.co/models)

**Popular models:**
- **Text:** GPT-2, BERT, T5, LLaMA
- **Image:** Stable Diffusion, DALL-E mini
- **Audio:** Whisper, Wav2Vec2
- **Multimodal:** CLIP, BLIP

### Download Models

```python
from transformers import AutoModel, AutoTokenizer

# Automatically downloads and caches
model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

**Cache location:**
- Linux/Mac: `~/.cache/huggingface/`
- Windows: `C:\Users\<username>\.cache\huggingface\`

### Upload Models (Optional)

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
 path_or_fileobj="model.bin",
 path_in_repo="model.bin",
 repo_id="username/model-name",
 token="your_token"
)
```

## Common Tasks

### Text Generation

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

input_ids = tokenizer.encode("Hello, I am", return_tensors='pt')
output = model.generate(input_ids, max_length=50)
print(tokenizer.decode(output[0]))
```

### Question Answering

```python
qa_pipeline = pipeline('question-answering')

context = "HuggingFace is a company that provides NLP tools."
question = "What does HuggingFace provide?"

result = qa_pipeline(question=question, context=context)
print(result['answer'])
```

### Translation

```python
translator = pipeline('translation_en_to_fr')
result = translator("Hello, how are you?")
print(result[0]['translation_text'])
```

### Summarization

```python
summarizer = pipeline('summarization')

text = """
Long article text here...
"""

summary = summarizer(text, max_length=130, min_length=30)
print(summary[0]['summary_text'])
```

## Advanced Features

### Custom Models

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
 'bert-base-uncased',
 num_labels=3 # Custom number of classes
)
```

### Fine-tuning

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
 output_dir='./results',
 num_train_epochs=3,
 per_device_train_batch_size=16,
 save_steps=10_000,
)

trainer = Trainer(
 model=model,
 args=training_args,
 train_dataset=train_dataset,
 eval_dataset=eval_dataset,
)

trainer.train()
```

### Datasets

```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset('imdb')

print(dataset['train'][0])
# {'text': '...', 'label': 1}
```

## GPU Acceleration

### Single GPU

```python
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

### Multiple GPUs

```python
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, dataloader = accelerator.prepare(
 model, optimizer, dataloader
)
```

## Authentication

### Create Account

1. Visit [huggingface.co](https://huggingface.co)
2. Sign up (free)

### Get Access Token

1. Go to Settings → Access Tokens
2. Create new token
3. Copy token

### Login

```bash
huggingface-cli login
```

Or in Python:
```python
from huggingface_hub import login
login(token="your_token")
```

## Best Practices

1. **Cache models** - Don't download repeatedly
2. **Use appropriate model size** - Smaller for CPU, larger for GPU
3. **Batch processing** - Process multiple inputs together
4. **Monitor memory** - Large models require significant RAM/VRAM
5. **Read model cards** - Understand model capabilities and limitations

## Troubleshooting

### Issue: Out of memory

**Solution:**
```python
# Use smaller model
model = AutoModel.from_pretrained('distilbert-base-uncased')

# Or use CPU
device = 'cpu'

# Or reduce batch size
batch_size = 1
```

### Issue: Slow download

**Solution:**
```python
# Download once, use cached version
from transformers import AutoModel
model = AutoModel.from_pretrained('bert-base-uncased')
# Subsequent loads are instant
```

### Issue: Model not found

**Solution:**
- Check model name on HuggingFace Hub
- Ensure internet connection
- Check for typos

## Resources

- [HuggingFace Documentation](https://huggingface.co/docs)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Datasets Documentation](https://huggingface.co/docs/datasets)
- [Course](https://huggingface.co/course) - Free NLP course
- [Forum](https://discuss.huggingface.co/) - Community support

## Example: Complete Workflow

```python
# 1. Install
!pip install transformers datasets

# 2. Import
from transformers import pipeline

# 3. Load model
classifier = pipeline('sentiment-analysis')

# 4. Use model
texts = [
 "I love this!",
 "This is terrible.",
 "It's okay."
]

for text in texts:
 result = classifier(text)[0]
 print(f"{text}: {result['label']} ({result['score']:.2%})")

# 5. Try different task
generator = pipeline('text-generation', model='gpt2')
output = generator("The future of AI is", max_length=50)
print(output[0]['generated_text'])
```

---

**Happy Modeling with HuggingFace! **
