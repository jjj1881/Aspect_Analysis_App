# Customer Review Intelligence Dashboard

This Streamlit app analyzes restaurant customer reviews using two fine-tuned Hugging Face models.

## Pipelines

1. Aspect Detection Pipeline
   - Input: review text
   - Output: detected business aspects

2. Aspect Sentiment Classification Pipeline
   - Input: review text + detected aspect
   - Output: positive / neutral / negative

## Hugging Face Models

- Aspect Detection: `jiangzy1881/aspect-detection-model`
- Aspect Sentiment: `jiangzy1881/aspect-sentiment-model`

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

Deploy this repository on Streamlit Cloud. The app loads both fine-tuned models directly from Hugging Face Hub.
