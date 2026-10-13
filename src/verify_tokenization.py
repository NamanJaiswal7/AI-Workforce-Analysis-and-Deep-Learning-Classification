from transformers import XLMRobertaTokenizer

def verify_tokenization():
    model_name = "xlm-roberta-base"
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
    
    sample_lt = "Informacinių sistemų programų inžinerija"
    tokens_lt = tokenizer.tokenize(sample_lt)
    print(f"Lithuanian: '{sample_lt}'")
    print(f"Tokens: {tokens_lt}")
    
    sample_en = "Information systems software engineering"
    tokens_en = tokenizer.tokenize(sample_en)
    print(f"\nEnglish: '{sample_en}'")
    print(f"Tokens: {tokens_en}")

if __name__ == "__main__":
    verify_tokenization()
