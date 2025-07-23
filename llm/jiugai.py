import os
import re
import csv
import json
import torch
import argparse
from typing import List, Tuple
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
    ).to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer, model

def read_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def get_paper_title(content: str) -> str:
    lines = content.splitlines()
    for line in lines:
        if line.strip():
            return line.strip()
    return "Untitled"

#############################
# Format detection and segmentation

def detect_reference_format(references_text: str) -> str:
    """
    bracketed: [1] ...
    parenthesized: (1) ...
    numberdot: 1. ... or 1) ...
    no index
    """
    lines = references_text.splitlines()
    
    bracketed_count = 0
    parenthesized_count = 0
    numberdot_count = 0
    
    for line in lines:
        line_stripped = line.strip()
        # [1]
        if re.match(r'^\[\d+\]\s', line_stripped):
            bracketed_count += 1
        # (1)
        elif re.match(r'^\(\d+\)\s', line_stripped):
            parenthesized_count += 1
        # 1. or 1)
        elif re.match(r'^\d+(\.|\))\s', line_stripped):
            numberdot_count += 1

    if bracketed_count >= 3:
        return "bracketed"
    elif parenthesized_count >= 3:
        return "parenthesized"
    elif numberdot_count >= 3:
        return "numberdot"
    else:
        return "none"

def parse_bracketed_references(references_text: str) -> list:
    """
    [1], [2]
    """
    raw_refs = re.split(r'(?=^\[\d+\])', references_text, flags=re.MULTILINE)
    references = []
    for ref_chunk in raw_refs:
        ref_chunk = ref_chunk.strip()
        if ref_chunk:
            references.append(ref_chunk)
    return references

def parse_parenthesized_references(references_text: str) -> list:
    """
    (1), (2) 
    """
    raw_refs = re.split(r'(?=^\(\d+\))', references_text, flags=re.MULTILINE)
    references = []
    for ref_chunk in raw_refs:
        ref_chunk = ref_chunk.strip()
        if ref_chunk:
            references.append(ref_chunk)
    return references

def parse_numberdot_references(references_text: str) -> list:
    """
    1. or 1)
    """
    raw_refs = re.split(r'(?=^\d+(\.|\))\s)', references_text, flags=re.MULTILINE)
    references = []
    for ref_chunk in raw_refs:
        ref_chunk = ref_chunk.strip()
        if ref_chunk:
            references.append(ref_chunk)
    return references

def heuristic_line_split_references(references_text: str) -> list:
    lines = references_text.strip().splitlines()
    references = []
    current_ref_lines = []

    def is_new_reference_line(line: str) -> bool:
        if not line:
            return False
        return line[0].isupper() or line[0].isdigit()

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue  # Skip blank line

        if current_ref_lines and is_new_reference_line(line_stripped):
            full_ref = " ".join(current_ref_lines).strip()
            if len(full_ref) > 5:
                references.append(full_ref)
            current_ref_lines = [line_stripped]
        else:
            current_ref_lines.append(line_stripped)

    if current_ref_lines:
        full_ref = " ".join(current_ref_lines).strip()
        if len(full_ref) > 5:
            references.append(full_ref)
    return references

def parse_noindex_references(references_text: str) -> list:
    """
    no index
    """
    raw_refs = re.split(r'\n\s*\n+', references_text.strip())
    references = []
    for ref_chunk in raw_refs:
        ref_chunk = ref_chunk.strip()
        if len(ref_chunk) > 5:
            lines_in_chunk = ref_chunk.splitlines()
            one_ref = " ".join(l.strip() for l in lines_in_chunk if l.strip())
            references.append(one_ref)

    if len(references) <= 1:
        # fallback
        references = heuristic_line_split_references(references_text)
    return references

def parse_references(references_text: str) -> list:
    # remove "References"
    lines = references_text.strip().split("\n", 1)
    if len(lines) > 1:
        if re.match(r'(?i)^\s*references\s*$', lines[0]):
            references_text = lines[1]

    fmt = detect_reference_format(references_text)
    if fmt == "bracketed":
        references = parse_bracketed_references(references_text)
    elif fmt == "parenthesized":
        references = parse_parenthesized_references(references_text)
    elif fmt == "numberdot":
        references = parse_numberdot_references(references_text)
    else:
        references = parse_noindex_references(references_text)

    clean_refs = []
    for ref in references:
        r = ref.strip()
        # remove "[1]", "(1)", "1.", "1)" ...
        r = re.sub(r'^(?:\[\d+\]|\(\d+\)|\d+\.|\d+\))\s*', '', r)
        clean_refs.append(r)

    final_refs = [r for r in clean_refs if len(r) > 5]
    return final_refs

####################

def extract_references(content: str) -> str:
    references = ""
    lines = content.splitlines()
    ref_start = None
    keywords = [r"References"]
    for i, line in enumerate(lines):
        if any(re.search(kw, line, re.IGNORECASE) for kw in keywords):
            ref_start = i
            break

    if ref_start is None:
        print("No References section found in the text.")
        return ""

    possible_stops = [r'\bAppendix\b', 
                      r'\bAPPENDIX\b', 
                      r'\bNotation\b', 
                      r'\bNOTATION\b', 
                      r'\bSupplementary\b', 
                      r'\bSUPPLEMENTARY\b']
                    #   r'^[A-Z]\.\s', 
                    #   r'^[A-Z]\s']
    
    ref_end = len(lines)
    for j in range(ref_start+1, len(lines)):
        for stop_word in possible_stops:
            if re.search(stop_word, lines[j], re.IGNORECASE):
                ref_end = j
                break
        if ref_end != len(lines):
            break
    
    references_lines = lines[ref_start:ref_end]
    references = "\n".join(references_lines)

    return references

def extract_main_content(content: str) -> str:
    main_content = ""
    lines = content.splitlines()
    ref_start = None
    # keywords = [r"Abstract", r"Introduction"]
    keywords = [r"References"]
    for i, line in enumerate(lines):
        if any(re.search(kw, line, re.IGNORECASE) for kw in keywords):
            ref_start = i
            break
    if ref_start is not None:
        main_content = "\n".join(lines[:ref_start])
    else:
        main_content = content
    return main_content

def summarize_content(model, tokenizer, content: str, device="cuda", max_new_tokens=300) -> str:

    system_prompt = (
        "You are a helpful assistant for academic summarization. "
        "Do not restate the entire text; provide a concise summary."
    )
    user_prompt = (
        "Please write a thorough, multi-paragraph summary covering" 
        "the main contributions, methodologies, experimental results," 
        "and conclusions. The summary should be around 1000 words.\n\n"
        f"{content}\n\n"
        "Your summary should be relatively short and must not copy the text verbatim."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
        padding=True
    ).to(device)

    # if DataParallel: use model.module to generate
    gen_model = model.module if hasattr(model, 'module') else model

    outputs = gen_model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        # temperature=0.6,
        # top_p=0.9,
        # repetition_penalty=1.2,
        pad_token_id=tokenizer.pad_token_id
    )

    generated = outputs[0][input_ids.shape[1]:]
    summary = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return summary


def select_and_rank_references(
    model, 
    tokenizer, 
    main_content: str, 
    references: List[str], 
    device="cuda"
) -> List[Tuple[int, str]]:
    max_main_content_length = 2000
    if len(main_content) > max_main_content_length:
        main_content = summarize_content(model, tokenizer, main_content, device=device)
        print(f"Content is too long, using summarized main content:\n{main_content[:500]}...\n")
    references_text = "\n".join([f"[{i+1}] {ref}" for i, ref in enumerate(references)])
    
    prompt = (
        "The following is the main content of a research paper and its list of references. "
        "Please analyze the importance of each reference based on the main content, "
        "select the five most important references, and rank them in order of importance. "
        "Output the selected references strictly in the following format with no additional text:\n\n"
        "Main content:\n"
        f"{main_content}\n\n"
        "References list:\n"
        f"{references_text}\n\n"
        "Rank:\n"
    )
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=2048, 
        padding=True
    ).to(device)
    
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=1000,
        # temperature=0.8,
        # top_p=0.8,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.pad_token_id,
        num_return_sequences=1,
        do_sample=False
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("LLM Raw Response:\n", response, "\n")
    selected_refs = []
    for line in response.splitlines():
        line = line.strip()
        # "[1] Reference" or "(1) Reference" or "1. Reference" ? 
        match = re.match(r"[\(\[]?(\d+)[\)\]]?[\.]?\s+(.*)", line)
        if match:
            index = int(match.group(1))
            if 1 <= index <= len(references):
                ref_text = references[index-1]
                selected_refs.append((index, ref_text))
            if len(selected_refs) == 5:  
                break
    return selected_refs

def save_to_csv(selected_refs: List[Tuple[int, str]], output_file="selected_references.csv"):
    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Rank", "Reference"])
        for rank, (index, ref) in enumerate(selected_refs, start=1):
            writer.writerow([rank, ref])

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pdf_path = "2212.06872v5.pdf"
    print(f"Reading PDF: {pdf_path}")
    content = read_pdf(pdf_path)
    if not content.strip():
        print("PDF content is empty.")
        return
    title = get_paper_title(content)
    print(f"Paper Title (heuristic): {title}")
    main_content = extract_main_content(content)
    references_text = extract_references(content)
    if not references_text.strip():
        print("No valid references found.")
        return
    references = parse_references(references_text)
    print(f"Total {len(references)} references found.")
    if len(references) < 5:
        print("References are fewer than 5, skipping ranking.")
        return
    print("Loading llama-3-8b...")
    tokenizer, model = load_model(device=device)
    
    print("Selecting and ranking the five most important references using LLM...")
    selected_refs = select_and_rank_references(model, tokenizer, main_content, references, device=device)
    if len(selected_refs) < 5:
        print("Failed to select five references.")
    else:
        print("Successfully selected five references.")
        save_to_csv(selected_refs, output_file="selected_references.csv")
        print(f"Results saved to 'selected_references'.\n")
        print("Top 5 References:")
        for rank, (index, ref) in enumerate(selected_refs, start=1):
            print(f"{rank}. [Index={index}] {ref}")

            
if __name__ == "__main__":
    
    main()


