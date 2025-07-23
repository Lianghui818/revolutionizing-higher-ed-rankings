import os
import re
import csv
import glob
import torch
import argparse
from typing import List, Tuple
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer, model

def read_pdf(file_path: str) -> str:
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        print(f"Error reading PDF {file_path}: {str(e)}")
        return ""

def get_paper_title(content: str) -> str:
    lines = content.splitlines()
    for line in lines:
        if line.strip():
            return line.strip()
    return "Untitled"

#####
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
    pattern = r"\.\s*\n\s*\n+(?=[A-Z][a-zA-Z-]+ [A-Z]|[A-Z]\. [A-Z][a-zA-Z-]+)"

    # raw_refs = re.split(r'\.\s*\n\s*\n+', references_text.strip())
    # raw_refs = [ref.strip() + '.' for ref in raw_refs if ref.strip()]
    raw_refs = re.split(pattern, references_text.strip())
    raw_refs = [ref.strip() for ref in raw_refs if ref.strip()]

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
    # final_refs = []
    # for idx, ref in enumerate(clean_refs):
    #     if len(ref) > 5:
    #         numbered_ref = f"[{idx+1}] {ref}"
    #         final_refs.append(numbered_ref)

    return final_refs
#####

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
                      r'\bSUPPLEMENTARY\b',
                      r'\bAttention Visualizations\b',
                      r'\bA NOTATIONS\b']
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
        temperature=0.6,
        top_p=0.9,
        repetition_penalty=1.2,
        use_cache=True,
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
    
    # 简化prompt，减少幻觉风险
    prompt = (
        "Based on the following research paper content and references list, "
        "select the 5 most important references and rank them by importance.\n\n"
        "IMPORTANT: Only respond with a numbered list using this exact format:\n"
        "1. [X]\n"
        "2. [Y]\n"
        "3. [Z]\n"
        "4. [A]\n"
        "5. [B]\n"
        "Where X, Y, Z, A, B are reference numbers from the list below.\n\n"
        "Paper content:\n"
        f"{main_content}\n\n"
        "References:\n"
        f"{references_text}\n\n"
        "Top 5 most important references:"
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
        max_new_tokens=200,  # 减少输出长度，避免幻觉
        repetition_penalty=1.2,
        pad_token_id=tokenizer.pad_token_id,
        num_return_sequences=1,
        use_cache=True,
        do_sample=False        
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("LLM Raw Response:\n", response, "\n")
    
    # 从response中提取LLM生成的部分（去掉prompt）
    prompt_in_response = prompt.split("Top 5 most important references:")[0] + "Top 5 most important references:"
    if prompt_in_response in response:
        llm_output = response.split(prompt_in_response)[-1].strip()
    else:
        llm_output = response.strip()
    
    print("LLM Output (cleaned):\n", llm_output, "\n")
    
    selected_refs = []
    
    # 更严格的解析：只匹配标准格式
    list_pattern = r"(?:^|\n)\s*(\d+)\.?\s*\[(\d+)\]"
    list_matches = re.findall(list_pattern, llm_output)
    
    print("Extracted matches:", list_matches)
    
    for rank_str, ref_idx_str in list_matches:
        try:
            rank = int(rank_str)
            ref_idx = int(ref_idx_str)
            
            if 1 <= ref_idx <= len(references):
                
                if ref_idx not in [idx for idx, _ in selected_refs]:
                    ref_text = references[ref_idx-1]
                    selected_refs.append((ref_idx, ref_text))
                    print(f"Added reference {ref_idx}: {ref_text[:100]}...")
                else:
                    print(f"Reference {ref_idx} already selected, skipping")
            else:
                print(f"Invalid reference index {ref_idx} (should be 1-{len(references)})")
        except ValueError:
            print(f"Failed to parse rank/index: {rank_str}, {ref_idx_str}")
    
    if len(selected_refs) < 5:
        print("Not enough references found with strict matching, trying loose matching...")
        
        
        bracket_pattern = r"\[(\d+)\]"
        all_matches = re.findall(bracket_pattern, llm_output)
        
        for ref_idx_str in all_matches:
            if len(selected_refs) >= 5:
                break
                
            try:
                ref_idx = int(ref_idx_str)
                if 1 <= ref_idx <= len(references):
                    if ref_idx not in [idx for idx, _ in selected_refs]:
                        ref_text = references[ref_idx-1]
                        selected_refs.append((ref_idx, ref_text))
                        print(f"Added reference {ref_idx} (loose match): {ref_text[:100]}...")
            except ValueError:
                continue
    
    if len(selected_refs) < 5:
        print("Still not enough references, filling with first available ones...")
        for i in range(len(references)):
            if len(selected_refs) >= 5:
                break
            ref_idx = i + 1
            if ref_idx not in [idx for idx, _ in selected_refs]:
                ref_text = references[ref_idx-1]
                selected_refs.append((ref_idx, ref_text))
                print(f"Added reference {ref_idx} (fallback): {ref_text[:100]}...")
    
    # 验证最终结果
    final_refs = selected_refs[:5]
    print(f"\nFinal selected references ({len(final_refs)}):")
    for i, (ref_idx, ref_text) in enumerate(final_refs, 1):
        print(f"{i}. [Index={ref_idx}] {ref_text[:100]}...")
    
    return final_refs 


def save_to_csv(selected_refs: List[Tuple[int, str]], output_file: str):
    """保存选中的参考文献到CSV文件"""
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Rank", "Reference"])
        for rank, (index, ref) in enumerate(selected_refs, start=1):
            ref_single_line = re.sub(r'\s+', ' ', ref).strip()
            writer.writerow([rank, ref_single_line])

def append_to_combined_csv(paper_name: str, selected_refs: List[Tuple[int, str]], output_file: str):
    """将单篇论文的参考文献追加到汇总CSV文件中"""
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 检查文件是否存在，如果不存在则创建并写入表头
    file_exists = os.path.exists(output_file)
    
    with open(output_file, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        
        # 如果是新文件，写入表头
        if not file_exists:
            writer.writerow(["Paper_Name", "Rank", "Reference"])
        
        # 写入该论文的所有参考文献
        for rank, (index, ref) in enumerate(selected_refs, start=1):
            ref_single_line = re.sub(r'\s+', ' ', ref).strip()
            writer.writerow([paper_name, rank, ref_single_line])

def process_single_pdf(pdf_path: str, model, tokenizer, device="cuda"):
    """处理单个PDF文件，返回参考文献结果"""
    try:
        print(f"\nProcessing: {pdf_path}")
        content = read_pdf(pdf_path)
        if not content.strip():
            print(f"PDF content is empty: {pdf_path}")
            return None
        
        title = get_paper_title(content)
        print(f"Paper Title: {title}")
        
        main_content = extract_main_content(content)
        references_text = extract_references(content)
        
        if not references_text.strip():
            print(f"No valid references found: {pdf_path}")
            return None
        
        references = parse_references(references_text)
        print(f"Total {len(references)} references found.")
        
        if len(references) < 5:
            print(f"References are fewer than 5, skipping: {pdf_path}")
            return None
        
        print("Selecting and ranking the five most important references...")
        selected_refs = select_and_rank_references(model, tokenizer, main_content, references, device=device)
        
        if len(selected_refs) < 5:
            print(f"Failed to select five references: {pdf_path}")
            return None
        
        print("\nVerified Top 5 References:")
        verified_refs = []
        for rank, (index, ref) in enumerate(selected_refs, start=1):
            
            if 1 <= index <= len(references):
                actual_ref = references[index-1]
                
                if ref == actual_ref:
                    verified_refs.append((index, ref))
                    ref_single_line = re.sub(r'\s+', ' ', ref).strip()
                    print(f"{rank}. [Index={index}] {ref_single_line[:100]}...")
                else:
                    print(f"WARNING: Reference mismatch at index {index}")
                    print(f"Expected: {actual_ref[:100]}...")
                    print(f"Got: {ref[:100]}...")
                    # 使用正确的参考文献
                    verified_refs.append((index, actual_ref))
                    ref_single_line = re.sub(r'\s+', ' ', actual_ref).strip()
                    print(f"{rank}. [Index={index}] (CORRECTED) {ref_single_line[:100]}...")
            else:
                print(f"ERROR: Invalid index {index} for reference list of length {len(references)}")
                return None
        
        if len(verified_refs) != 5:
            print(f"Verification failed: expected 5 references, got {len(verified_refs)}")
            return None
        
        return verified_refs
        
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
        return None

def find_cvpr_folders(downloads_path: str) -> List[str]:

    print(f"Searching for CVPR folders in: {downloads_path}")

    patterns = [
        os.path.join(downloads_path, "CVPR202*"),
        os.path.join(downloads_path, "cvpr202*"),  # 小写
        os.path.join(downloads_path, "*CVPR202*"), # 包含CVPR的文件夹
    ]
    
    all_folders = set()
    for pattern in patterns:
        folders = glob.glob(pattern)
        all_folders.update(folders)
    
    cvpr_folders = [folder for folder in all_folders if os.path.isdir(folder)]
    cvpr_folders.sort()  # 按名称排序
    
    print(f"Found {len(cvpr_folders)} potential CVPR folders:")
    for folder in cvpr_folders:
        pdf_count = len(glob.glob(os.path.join(folder, "*.pdf")))
        print(f"  - {folder} ({pdf_count} PDF files)")
    
    return cvpr_folders

def find_iccv_folders(downloads_path: str) -> List[str]:
    print(f"Searching for ICCV folders in: {downloads_path}")
    
    patterns = [
        os.path.join(downloads_path, "ICCV202*"),
        os.path.join(downloads_path, "iccv202*"),  # 小写
        os.path.join(downloads_path, "*ICCV202*"), # 包含CVPR的文件夹
    ]
    
    all_folders = set()
    for pattern in patterns:
        folders = glob.glob(pattern)
        all_folders.update(folders)
    
    iccv_folders = [folder for folder in all_folders if os.path.isdir(folder)]
    iccv_folders.sort()  # 按名称排序
    
    print(f"Found {len(iccv_folders)} potential ICCV folders:")
    for folder in iccv_folders:
        pdf_count = len(glob.glob(os.path.join(folder, "*.pdf")))
        print(f"  - {folder} ({pdf_count} PDF files)")
    
    return iccv_folders

def batch_process_cvpr_papers(downloads_path: str = "Downloads", references_path: str = "References", device: str = "cuda"):

    if not os.path.exists(downloads_path):
        print(f"Downloads folder not found: {downloads_path}")
        return
    
    cvpr_folders = find_cvpr_folders(downloads_path)
    if not cvpr_folders:
        print(f"No CVPR202X_xxx folders found in {downloads_path}")
        return
    
    print(f"\nFound {len(cvpr_folders)} CVPR folders:")
    for folder in cvpr_folders:
        print(f"  - {folder}")
    
    print("\nLoading DeepSeek model...")
    tokenizer, model = load_model(device=device)
    
    if not os.path.exists(references_path):
        os.makedirs(references_path)
    
    total_processed = 0
    total_successful = 0
    total_folders_processed = 0
    
    for cvpr_folder in cvpr_folders:
        folder_name = os.path.basename(cvpr_folder)
        print(f"\n{'='*60}")
        print(f"Processing folder {total_folders_processed + 1}/{len(cvpr_folders)}: {folder_name}")
        print(f"{'='*60}")
        
        try:
            
            combined_csv_file = os.path.join(references_path, f"{folder_name}_re.csv")
            
            if os.path.exists(combined_csv_file):
                os.remove(combined_csv_file)
                print(f"Removed existing file: {combined_csv_file}")
            
            pdf_pattern = os.path.join(cvpr_folder, "*.pdf")
            pdf_files = glob.glob(pdf_pattern)
            
            if not pdf_files:
                print(f"No PDF files found in {cvpr_folder}")
                total_folders_processed += 1
                continue
            
            print(f"Found {len(pdf_files)} PDF files in {folder_name}")
            print(f"Combined results will be saved to: {combined_csv_file}")
            
            folder_processed = 0
            folder_successful = 0
            
            for i, pdf_file in enumerate(pdf_files, 1):
                pdf_basename = os.path.splitext(os.path.basename(pdf_file))[0]
                
                print(f"\n--- Processing PDF {i}/{len(pdf_files)}: {pdf_basename} ---")
                
                folder_processed += 1
                total_processed += 1
                
                selected_refs = process_single_pdf(pdf_file, model, tokenizer, device)
                
                if selected_refs is not None and len(selected_refs) == 5:
            
                    print(f"Successfully extracted {len(selected_refs)} references from {pdf_basename}")
                    
                    append_to_combined_csv(pdf_basename, selected_refs, combined_csv_file)
                    print(f"Added {pdf_basename} references to combined CSV")
                    folder_successful += 1
                    total_successful += 1
                else:
                    print(f"Failed to process: {pdf_basename} (extracted {len(selected_refs) if selected_refs else 0} references)")
            
            total_folders_processed += 1
            
            print(f"\nFolder {folder_name} summary:")
            print(f"  Processed: {folder_processed} PDFs")
            print(f"  Successful: {folder_successful} PDFs")
            print(f"  Failed: {folder_processed - folder_successful} PDFs")
            if folder_successful > 0:
                print(f"  Total references in CSV: {folder_successful * 5}")
                print(f"  Combined CSV saved as: {combined_csv_file}")
            
        except Exception as e:
            print(f"Error processing folder {folder_name}: {str(e)}")
            total_folders_processed += 1
            continue
    
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total folders found: {len(cvpr_folders)}")
    print(f"Total folders processed: {total_folders_processed}")
    print(f"Total PDFs processed: {total_processed}")
    print(f"Total PDFs successful: {total_successful}")
    print(f"Total PDFs failed: {total_processed - total_successful}")
    print(f"Total references extracted: {total_successful * 5}")
    print(f"Results saved in: {references_path}")
    
    created_csvs = [f for f in os.listdir(references_path) if f.endswith('_re.csv')]
    print(f"Number of combined CSV files created: {len(created_csvs)}")
    
    if len(created_csvs) > 0:
        print("Created files:")
        for csv_file in sorted(created_csvs):
            csv_path = os.path.join(references_path, csv_file)
            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    line_count = sum(1 for line in f) - 1  # 减去表头
                print(f"  - {csv_file}: {line_count} references")
            except Exception as e:
                print(f"  - {csv_file}: Error reading file ({str(e)})")
    
    processed_folder_names = [os.path.splitext(f)[0].replace('_re', '') for f in created_csvs]
    original_folder_names = [os.path.basename(folder) for folder in cvpr_folders]
    skipped_folders = [name for name in original_folder_names if name not in processed_folder_names]
    
    if skipped_folders:
        print(f"\nWARNING: The following folders were completely skipped:")
        for folder in skipped_folders:
            print(f"  - {folder}")
        print("This might be due to:")
        print("  - No PDF files in the folder")
        print("  - All PDFs failed to process")
        print("  - Processing errors")

def debug_folder_structure(downloads_path: str = "Downloads"):
    print(f"Debugging folder structure in: {downloads_path}")
    
    if not os.path.exists(downloads_path):
        print(f"ERROR: Downloads folder does not exist: {downloads_path}")
        return
    
    all_items = os.listdir(downloads_path)
    all_folders = [item for item in all_items if os.path.isdir(os.path.join(downloads_path, item))]
    
    print(f"\nAll folders in {downloads_path}:")
    for folder in sorted(all_folders):
        folder_path = os.path.join(downloads_path, folder)
        pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
        is_cvpr = folder.startswith("CVPR202") or folder.startswith("cvpr202")
        print(f"  - {folder} ({'CVPR' if is_cvpr else 'NOT CVPR'}) - {len(pdf_files)} PDFs")
    
    cvpr_folders = [f for f in all_folders if "cvpr202" in f.lower()]
    print(f"\nCVPR-like folders found: {len(cvpr_folders)}")
    for folder in sorted(cvpr_folders):
        folder_path = os.path.join(downloads_path, folder)
        pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
        print(f"  - {folder}: {len(pdf_files)} PDF files")
        if len(pdf_files) > 0:
            print(f"    Sample PDFs: {[os.path.basename(f) for f in pdf_files[:3]]}")

def main():
    parser = argparse.ArgumentParser(description="Batch process CVPR papers for reference extraction")
    parser.add_argument("--downloads_path", default="Downloads", help="Path to Downloads folder containing CVPR folders")
    parser.add_argument("--references_path", default="References", help="Path to output References folder")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda/cpu)")
    parser.add_argument("--single_pdf", help="Process a single PDF file (for testing)")
    parser.add_argument("--debug", action="store_true", help="Debug folder structure without processing")
    
    args = parser.parse_args()
    
    if args.debug:
        debug_folder_structure(args.downloads_path)
        return
    
    if args.single_pdf:
        if not os.path.exists(args.single_pdf):
            print(f"PDF file not found: {args.single_pdf}")
            return
        
        print("Loading DeepSeek model...")
        tokenizer, model = load_model(device=args.device)
        
        pdf_basename = os.path.splitext(os.path.basename(args.single_pdf))[0]
        output_file = f"{pdf_basename}_references.csv"
        
        selected_refs = process_single_pdf(args.single_pdf, model, tokenizer, args.device)
        if selected_refs is not None:
            save_to_csv(selected_refs, output_file)
            print(f"Successfully processed: {args.single_pdf}")
            print(f"Results saved to: {output_file}")
        else:
            print(f"Failed to process: {args.single_pdf}")
    else:
        batch_process_cvpr_papers(args.downloads_path, args.references_path, args.device)

if __name__ == "__main__":
    main()


# python t2.py --downloads_path ~/hpc-share/revolutionizing-higher-ed-rankings/Downloads/ --references_path ~/hpc-share/revolutionizing-higher-ed-rankings/References/