import pandas as pd
import json
import torch
import os
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def setup_model(model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", device_map="auto"):
    """
    Load the tokenizer and model from Hugging Face Hub.
    Uses half precision (float16) and automatic device mapping (e.g., GPU if available).
    """
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device_map
    )
    return tokenizer, model

def create_extraction_prompt(reference):
    """
    Create a prompt string asking the model to extract authors and title from a citation.
    The prompt instructs the model to output a JSON with two fields: authors (list) and title (string).
    """
    return f"""Extract the authors and title from this reference citation.
Reference: {reference}

Format your answer strictly as a valid JSON with two fields:
1. "authors": containing only the authors' names. Each author's name is a separate element in the list.
2. "title": containing only the paper title

"""

def extract_json_from_response(response):
    """
    Attempt to extract a JSON object with "authors" and "title" fields from the model's response.
    Uses regex to find JSON-like substrings and attempts to parse them.
    Falls back to simpler regex extraction or line scanning if needed.
    Returns a dict with extracted data or a failure flag.
    """
    json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
    matches = re.finditer(json_pattern, response)
    
    for match in matches:
        try:
            json_str = match.group(0)
            data = json.loads(json_str)
            
            if "authors" in data and "title" in data:
                return data
        except json.JSONDecodeError:
            continue
    
    authors_pattern = r'"authors"\s*:\s*"([^"]*)"'
    title_pattern = r'"title"\s*:\s*"([^"]*)"'
    
    authors_match = re.search(authors_pattern, response)
    title_match = re.search(title_pattern, response)
    
    if authors_match and title_match:
        return {
            "authors": authors_match.group(1),
            "title": title_match.group(1)
        }
    
    try:
        lines = response.strip().split('\n')
        for line in lines:
            if '{' in line and '}' in line and '"authors"' in line and '"title"' in line:

                json_str = line[line.find('{'):line.rfind('}')+1]
                try:
                    data = json.loads(json_str)
                    if "authors" in data and "title" in data:
                        return data
                except:
                    pass
    except:
        pass
    
    print(f"Unable to extract JSON from the response: {response}")
    return {"authors": "", "title": "", "extraction_failed": True}

def process_csv(csv_file, output_file):
    """
    Process a CSV file with reference citations.
    For each citation, generate a prompt and run the model to extract authors and title.
    Save incremental results to a partial JSON file, and final results to the output file.
    """
    tokenizer, model = setup_model()

    try:
        df = pd.read_csv(csv_file)

        if df.shape[1] < 2:
            raise ValueError(f"The CSV file requires at least 2 columns")
        
        index_col = df.columns[1]
        ref_col = df.columns[2]
    
        
    except Exception as e:
        print(f"Error when reading csv file: {str(e)}")
        try:
            df = pd.read_csv(csv_file, header=None)
            df.columns = [f"col_{i}" for i in range(df.shape[2])]
            index_col = "col_1"
            ref_col = "col_2"
        except Exception as e2:
            print(f"Also Faild: {str(e2)}")
            raise
    
    results = []
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing reference"):
        try:
            reference = str(row[ref_col]).strip()
            # Skip blank
            if reference == "nan" or len(reference) < 10:
                continue
            
            prompt = create_extraction_prompt(reference)
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1000,
                    temperature=0.1,
                    top_p=0.95,
                    do_sample=False
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # print(f"\nPrompt:\n{prompt}")
            print(f"\nResponse:\n{response}")
            
            extracted_data = extract_json_from_response(response)
            
            extracted_data["original_index"] = row[index_col]
            
            results.append(extracted_data)
            
            print(f"Result: {json.dumps(extracted_data, ensure_ascii=False)}")
            with open(f"{output_file}_partial.json", "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            print(f"Error when processing on {row[index_col]} 's refernce: {str(e)}")
            results.append({
                "authors": "error",
                "title": "error",
                "original_index": row[index_col],
                "error": str(e)
            })
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Done. Result has been saved to {output_file}")

def main():
    """
    Main entry point of the script.
    Optionally prints GPU info (commented out).
    Sets input CSV and output JSON filenames.
    Calls the processing function.
    """
    # check gpus
    # print(f"available gpu: {torch.cuda.device_count()}")
    # for i in range(torch.cuda.device_count()):
    #     print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # input_csv = "selected_references.csv"  
    # output_json = "extracted_references.json"
    input_csv = "../References/ICCV2021_1013_re.csv"  
    output_json = "../AT/ICCV2021_1013_re.json"
    
    process_csv(input_csv, output_json)

if __name__ == "__main__":
    main()