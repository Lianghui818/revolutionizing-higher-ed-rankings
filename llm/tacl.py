import os
import re
import csv
import glob
import time
import json
import math
import torch
import argparse
import logging
import warnings
from typing import List, Tuple
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForCausalLM
import pdfplumber

# Silence noisy logs/warnings
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pdfplumber").setLevel(logging.ERROR)
logging.getLogger("pdfrw").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*gray level.*")
warnings.filterwarnings("ignore", message=".*FontBBox.*")

def load_model(model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32
    ).to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer, model

ACL_HEADER_PATTERNS = [
    r"Published as a conference paper at ACL 20\d{2}",
    r"OpenReview\.net",
    r"Under review as a conference paper at ACL 20\d{2}",
]

def read_pdf(file_path: str) -> str:
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
        return text
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
        return ""

def clean_text_common(s: str) -> str:
    s = s.replace("\r", "\n")
    for pat in ACL_HEADER_PATTERNS:
        s = re.sub(pat, " ", s, flags=re.IGNORECASE)
    s = re.sub(r"(\w)-\n(\w)", r"\1-\2", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def get_paper_title(pdf_path: str) -> str:
    try:
        meta = PdfReader(pdf_path).metadata
        if meta and getattr(meta, "title", None):
            t = str(meta.title).strip()
            if t and not t.lower().startswith("proceedings of"):
                return t
    except Exception:
        pass
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if not pdf.pages:
                return "Untitled"
            p0 = pdf.pages[0]
            chars = p0.chars or []
            if not chars:
                return "Untitled"
            upper_half = [c for c in chars if c.get("top", 0) < p0.height * 0.60 and c.get("text", "").strip()]
            pool = upper_half or chars
            max_size = max(c.get("size", 0) for c in pool)
            big = [c for c in pool if abs(c.get("size", 0) - max_size) <= 0.6]
            big.sort(key=lambda c: (round(c["top"], 1), c["x0"]))
            lines, cur_top, buf = [], None, []
            for ch in big:
                t = round(ch["top"], 1)
                if cur_top is None or abs(t - cur_top) > 1.6:
                    if buf:
                        lines.append("".join(x["text"] for x in buf).strip())
                    buf, cur_top = [ch], t
                else:
                    buf.append(ch)
            if buf:
                lines.append("".join(x["text"] for x in buf).strip())
            for ln in lines:
                s = ln.strip()
                if len(s.split()) >= 3 and not s.lower().startswith("proceedings of"):
                    return s
            txt = p0.extract_text(x_tolerance=1.2, y_tolerance=1.2) or ""
            parts = [x.strip() for x in txt.splitlines() if x.strip()]
            for i, ln in enumerate(parts):
                if re.fullmatch(r"[Aa]bstract", ln):
                    cand = sorted((parts[max(0, i-3):i]), key=len, reverse=True)
                    if cand:
                        s = cand[0]
                        if not s.lower().startswith("proceedings of"):
                            return s
            return "Untitled"
    except Exception:
        return "Untitled"

# Layout-aware reference extraction (Plan B
def join_line(line_words, x_gap_tol=1):
    line_words = sorted(line_words, key=lambda w: w["x0"])
    parts, prev_x1 = [], None
    for w in line_words:
        if prev_x1 is not None and (w["x0"] - prev_x1) > x_gap_tol:
            parts.append(" ")
        parts.append(w["text"])
        prev_x1 = w["x1"]
    return "".join(parts)

def lines_from_words(words, y_tol=3.2):
    if not words:
        return []

    x_min = min(w["x0"] for w in words)
    for w in words:
        w["_norm_x0"] = w["x0"] - x_min

    words = sorted(words, key=lambda w: (round(w["top"], 1), w["_norm_x0"]))

    lines, buf, cur_y = [], [], None
    for w in words:
        y = round(w["top"], 1)
        if cur_y is None or abs(y - cur_y) > y_tol:
            if buf:
                x0_min = min(x["_norm_x0"] for x in buf)
                avg_size = sum(x["size"] for x in buf) / len(buf)
                lines.append({"x0": x0_min, "text": join_line(buf), "size": avg_size})
            buf, cur_y = [w], y
        else:
            buf.append(w)
    if buf:
        x0_min = min(x["_norm_x0"] for x in buf)
        avg_size = sum(x["size"] for x in buf) / len(buf)
        lines.append({"x0": x0_min, "text": join_line(buf), "size": avg_size})
    return lines

def percentile(values, q):
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * q
    f = int(k)
    c = min(f + 1, len(v) - 1)
    if f == c:
        return v[f]
    return v[f] + (v[c] - v[f]) * (k - f)

def split_two_columns(words, page_width):
    if not words:
        return [words]
    centers = [(w["x0"] + w["x1"]) / 2 for w in words]
    m1, m2 = (page_width / 3.0), (page_width * 2 / 3.0)
    for _ in range(12):
        g1 = [c for c in centers if abs(c - m1) <= abs(c - m2)]
        g2 = [c for c in centers if abs(c - m1) > abs(c - m2)]
        if not g1 or not g2:
            break
        m1n, m2n = sum(g1) / len(g1), sum(g2) / len(g2)
        if abs(m1n - m1) < 0.8 and abs(m2n - m2) < 0.8:
            break
        m1, m2 = m1n, m2n
    thr = (m1 + m2) / 2.0
    gap = 10.0
    left = [w for w in words if (w["x0"] + w["x1"]) / 2 < thr - gap]
    right = [w for w in words if (w["x0"] + w["x1"]) / 2 > thr + gap]
    if not left or not right:
        return [words]
    return [left, right]

def infer_start_indent(lines):
    xs = [ln["x0"] for ln in lines]
    if not xs:
        return None
    c1, c2 = percentile(xs, 0.20), percentile(xs, 0.80)
    m1 = c1 if c1 is not None else min(xs)
    m2 = c2 if c2 is not None else max(xs)
    for _ in range(10):
        g1 = [x for x in xs if abs(x - m1) <= abs(x - m2)]
        g2 = [x for x in xs if abs(x - m1) > abs(x - m2)]
        if not g1 or not g2:
            break
        m1n, m2n = sum(g1) / len(g1), sum(g2) / len(g2)
        if abs(m1n - m1) < 0.5 and abs(m2n - m2) < 0.5:
            break
        m1, m2 = m1n, m2n
    start_mean = min(m1, m2)
    return start_mean + 4.0

AUTHOR_YEAR = re.compile(
    r"^[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'`-]+(?:[, ]+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'`.\-]+){0,2}"
    r"(?:\s+et al\.)?\.\s+(?:19|20)\d{2}[a-z]?\.\s"
)

MULTI_ANCHOR_SPLIT = re.compile(
    r"(?<!^)(?<=\.)\s+(?=[A-Z][^.\n]+?\.\s+(?:19|20)\d{2}[a-z]?\.\s)"
)

def segment_lines_by_indent(lines):
    if not lines:
        return []
    blocks, cur = [], []

    def is_valid_ref(ref):
      if ref.strip().startswith(("Figure:", "Table:")):
          return False

      first_words = re.match(r"^[A-ZÀ-ÖØ-Ý][\w.'-]*(?:\s+[A-ZÀ-ÖØ-Ý][\w.'-]*)+.*[,.]", ref.strip())
      return bool(first_words)

    x_start, x_indt = 0, 10 # magic numbers for now, will try to fix if I have time

    for ln in lines:
        x0 = int(ln["x0"])
        txt = ln["text"].strip()
        size = int(ln["size"])
        if not txt:
            continue
        if size >= 11: # hit a heading
            break
        if x0 == x_start:
            if cur:
              blocks.append(" ".join(cur))
            cur = [txt]
        elif (x0 <= int(1.2 * x_indt)) and (x0 >= int(0.8 * x_indt)):
            if cur[-1].endswith("-") and re.match(r"^[A-Za-z]", txt):
                cur[-1] = cur[-1][:-1] + txt
            else:
                cur.append(txt)

    if len(cur) > 1:
        blocks.append(" ".join(cur))

    split_blocks = []
    for b in blocks:
        b = re.sub(r"\s{2,}", " ", b).strip()
        if len(AUTHOR_YEAR.findall(b)) >= 2:
            parts = re.split(MULTI_ANCHOR_SPLIT, b)
            for p in parts:
                p = p.strip(" .")
                if p:
                    split_blocks.append(p)
        else:
            split_blocks.append(b.strip(" ."))

    filtered_blocks = [r for r in split_blocks if is_valid_ref(r)]

    final, seen = [], set()
    for r in filtered_blocks:
        r = re.sub(r"(https?://doi\.org/\S+)(\s+\1)+", r"\1", r)
        r = re.sub(r"(doi:\s*\S+)(\s+\1)+", r"\1", r, flags=re.I)
        key = re.sub(r"\W+", "", r.lower())[:220]
        if key in seen:
            continue
        seen.add(key)
        final.append(r)
    return final

def add_words_to_body(col_words, body):
    txt_parts = []
    sorted_words = sorted(col_words, key=lambda w: (w["top"], w["x0"]))
    text = " ".join(w["text"] for w in sorted_words)
    txt_parts.append(text)
    txt = " ".join(txt_parts)
    txt = re.sub(r"(?im)^proceedings of the .*acl.*\n?", "", txt)
    body.append(txt)
    return body

def extract_references_layout_and_main(pdf_path: str) -> Tuple[str, List[str]]:
    main_parts, refs_all, refs_lines = [], [], []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            ref_page_idx, ref_heading_bottom, ref_right_column = None, None, None
            for i, page in enumerate(pdf.pages):
                words = page.extract_words(x_tolerance=1, y_tolerance=1, use_text_flow=False, extra_attrs=["size"])
                # doesnt work sometimes?
                heading = [w for w in words if w["size"] > 11 and "reference" in w["text"].lower()]
                if heading:
                    ref_page_idx = i
                    ref_heading_bottom = max(w["bottom"] for w in heading)
                    mid_x = page.width / 2
                    heading_x = sum(w["x0"] for w in heading) / len(heading)
                    ref_right_column = heading_x > mid_x
                    break

            for i, page in enumerate(pdf.pages):
                if ref_page_idx is None or i < ref_page_idx:
                    words = page.extract_words(x_tolerance=1, y_tolerance=1, use_text_flow=False, extra_attrs=["size"])
                    cols = split_two_columns(words, page.width)
                    for col_words in cols:
                        main_parts = add_words_to_body(col_words, main_parts)

            if ref_page_idx is None:
                return ("\n".join(main_parts), [])
            for i in range(ref_page_idx, len(pdf.pages)):
                page = pdf.pages[i]
                words = page.extract_words(x_tolerance=1, y_tolerance=1, use_text_flow=False, extra_attrs=["size"])
                cols = split_two_columns(words, page.width)

                if i == ref_page_idx and ref_heading_bottom is not None:
                    left, right = cols
                    ref_col, other_col = (right, left) if ref_right_column else (left, right)

                    words_above_ref = [w for w in ref_col if w["top"] < ref_heading_bottom]
                    words_below_ref = [w for w in ref_col if w["top"] >= ref_heading_bottom + 1]

                    main_parts = add_words_to_body(words_above_ref, main_parts)
                    refs_lines.extend(lines_from_words(words_below_ref))
                    if not ref_right_column:
                        refs_lines.extend(lines_from_words(other_col))
                    else:
                        main_parts = add_words_to_body(other_col, main_parts)

                if i > ref_page_idx:
                    for col_words in cols:
                        cutoff = page.height * 0.92
                        col_words = [w for w in col_words if w["top"] < cutoff]
                        if not words:
                            continue
                        refs_lines.extend(lines_from_words(col_words))

        refs_all = segment_lines_by_indent(refs_lines)
        if len(refs_all) > 80:
            tmp = [re.sub(r"\s{2,}", " ", r).strip() for r in refs_all]
            def has_year(s): return re.search(r"(19|20)\d{2}", s) is not None
            merged = []
            for r in tmp:
                if len(r) < 70 or not has_year(r):
                    if merged:
                        merged[-1] = (merged[-1] + " " + r).strip()
                    else:
                        merged.append(r)
                else:
                    merged.append(r)
            refs_all = merged
        main_content = clean_text_common("\n".join(main_parts))
        return (main_content, refs_all)
    except Exception as e:
        print(f"[layout] Error extracting references from {pdf_path}: {e}")
        return ("", [])


def get_main_and_references(pdf_path: str, extractor: str = "layout") -> Tuple[str, List[str]]:
    if extractor == "layout":
        return extract_references_layout_and_main(pdf_path)
    else:
        content = read_pdf(pdf_path)
        txt = clean_text_common(content)
        m = re.search(r'(?is)\bReferences\b', txt)
        if not m:
            return (txt, [])
        refs_raw = txt[m.end():]
        refs = [p.strip() for p in re.split(r"\n{2,}", refs_raw) if p.strip()]
        return (txt[:m.end()], refs)


def summarize_content(model, tokenizer, content: str, device="cuda", max_new_tokens=3000) -> str:
    system_prompt = (
        "You are a helpful assistant for academic summarization. "
        "Do not restate the entire text; provide a concise summary."
    )
    user_prompt = (
        "Please write a concise multi-paragraph summary of the paper's main contributions, "
        "methods, experiments, and conclusions (around 400-600 words).\n\n"
        f"{content}\n\n"
        "Keep it objective and avoid copying sentences verbatim."
    )
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
        padding=True
    ).to(device)

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
    # print(references_text)

    prompt = (
       "Select the 5 most important references from the References list below based on the paper content. "
        "Output ONLY the 5 reference numbers in this exact format:\n"
        "1. [X]\n"
        "2. [Y]\n"
        "3. [Z]\n"
        "4. [A]\n"
        "5. [B]\n\n"
        "Where X, Y, Z, A, B are reference numbers from the References list below.\n\n"
        f"Paper content: {main_content}...\n\n"
        f"References:\n{references_text}\n\n"
        "Top 5 most important references:"
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant that selects the most important academic references. Respond only with the requested format."},
        {"role": "user", "content": prompt}
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    ).to(device)

    outputs = model.generate(
        inputs,
        max_new_tokens=4096,
        temperature=0.1,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.0,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True
    )

    generated_tokens = outputs[0][inputs.shape[1]:]

    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    print("LLM Response:", response)

    selected_refs = []
    pattern = r'(\d+)\.\s*\[(\d+)\]'
    matches = re.findall(pattern, response)
    for rank, ref_idx in matches:
        try:
            idx = int(ref_idx)
            if 1 <= idx <= len(references) and idx not in [i for i, _ in selected_refs]:
                selected_refs.append((idx, references[idx-1]))
                if len(selected_refs) >= 5:
                    break
        except:
            continue
    if len(selected_refs) < 5:
        numbers = re.findall(r'\[(\d+)\]', response)
        for num in numbers:
            try:
                idx = int(num)
                if 1 <= idx <= len(references) and idx not in [i for i, _ in selected_refs]:
                    selected_refs.append((idx, references[idx-1]))
                    if len(selected_refs) >= 5:
                        break
            except:
                continue
    if len(selected_refs) < 5:
        print("Using fallback selection...")
        for i in range(min(5, len(references))):
            if (i + 1) not in [idx for idx, _ in selected_refs]:
                selected_refs.append((i + 1, references[i]))
            if len(selected_refs) >= 5:
                break
    return selected_refs[:5]


def save_to_csv(selected_refs: List[Tuple[int, str]], output_file: str):
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
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_exists = os.path.exists(output_file)

    with open(output_file, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Paper_Name", "Rank", "Reference"])
        for rank, (index, ref) in enumerate(selected_refs, start=1):
            ref_single_line = re.sub(r'\s+', ' ', ref).strip()
            writer.writerow([paper_name, rank, ref_single_line])


def process_single_pdf(pdf_path: str, model, tokenizer, device="cuda", extractor="layout"):
    try:
        print(f"\nProcessing: {pdf_path}")
        title = get_paper_title(pdf_path)
        print(f"Paper Title: {title}")
        main_content, references = get_main_and_references(pdf_path, extractor=extractor)
        if not references:
            print("  [!] No references found")
            return None
        print(f"Total {len(references)} references found.")

        if len(references) < 5:
            print(f"References are fewer than 5, skipping: {pdf_path}")
            return None
        print("Selecting the five most important references...")

        selected_refs = select_and_rank_references(model, tokenizer, main_content, references, device=device)
        if len(selected_refs) < 5:
            print(f"Failed to select five references: {pdf_path}")
            return None
        print("\nVerified Top 5 References:")

        verified_refs = []
        for rank, (index, ref) in enumerate(selected_refs, start=1):
            if 1 <= index <= len(references):
                actual_ref = references[index - 1]
                if ref != actual_ref:
                    print(f"WARNING: Reference mismatch at index {index}")
                    print(f"Expected: {actual_ref[:100]}...")
                    print(f"Got: {ref[:100]}...")
                    ref = actual_ref
                verified_refs.append((index, ref))
                ref_single_line = re.sub(r'\s+', ' ', ref).strip()
                print(f"{rank}. [Index={index}] {ref_single_line[:2000]}...")
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

def find_acl_folders(downloads_path: str) -> List[str]:
    print(f"Searching for ACL folders in: {downloads_path}")
    patterns = [
        os.path.join(downloads_path, "ACL202*"),
        os.path.join(downloads_path, "acl202*"),
        os.path.join(downloads_path, "*ACL202*"),
    ]

    all_folders = set()

    for pattern in patterns:
        folders = glob.glob(pattern)
        all_folders.update(folders)

    acl_folders = [folder for folder in all_folders if os.path.isdir(folder)]
    acl_folders.sort()

    print(f"Found {len(acl_folders)} potential ACL folders:")

    for folder in acl_folders:
        pdf_count = len(glob.glob(os.path.join(folder, "*.pdf")))
        print(f"  - {folder} ({pdf_count} PDF files)")
    return acl_folders

def batch_process_acl_papers(downloads_path: str = "Downloads", references_path: str = "References", device: str = "cuda", extractor: str = "layout"):
    if not os.path.exists(downloads_path):
        print(f"Downloads folder not found: {downloads_path}")
        return

    acl_folders = find_acl_folders(downloads_path)

    if not acl_folders:
        print(f"No ACL202X_xxx folders found in {downloads_path}")
        return
    print(f"\nFound {len(acl_folders)} ACL folders:")

    for folder in acl_folders:
        print(f"  - {folder}")
    print("\nLoading DeepSeek model...")
    tokenizer, model = load_model(device=device)
    if not os.path.exists(references_path):
        os.makedirs(references_path)

    total_processed = 0
    total_successful = 0
    total_folders_processed = 0

    for acl_folder in acl_folders:
        folder_name = os.path.basename(acl_folder)
        print(f"\n{'='*60}")
        print(f"Processing folder {total_folders_processed + 1}/{len(acl_folders)}: {folder_name}")
        print(f"{'='*60}")
        try:
            combined_csv_file = os.path.join(references_path, f"{folder_name}_re.csv")
            if os.path.exists(combined_csv_file):
                os.remove(combined_csv_file)
                print(f"Removed existing file: {combined_csv_file}")
            pdf_pattern = os.path.join(acl_folder, "*.pdf")
            pdf_files = glob.glob(pdf_pattern)
            if not pdf_files:
                print(f"No PDF files found in {acl_folder}")
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
                selected_refs = process_single_pdf(
                    pdf_file, model, tokenizer, device,
                    extractor=extractor
                )
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
    print(f"Total folders found: {len(acl_folders)}")
    print(f"Total folders processed: {total_folders_processed}")
    print(f"Total PDFs processed: {total_processed}")
    print(f"Total PDFs successful: {total_successful}")
    print(f"Total PDFs failed: {total_processed - total_successful}")
    print(f"Total references extracted: {total_successful * 5}")
    print(f"Results saved in: {references_path}")
    created_csvs = [f for f in os.listdir(references_path) if f.endswith('_re.csv')]
    print(f"Number of combined CSV files created: {len(created_csvs)}")

    if len(created_csvs) > 0:
        for csv_file in sorted(created_csvs):
            csv_path = os.path.join(references_path, csv_file)
            try:
                _ = sum(1 for _ in open(csv_path, 'r', encoding='utf-8')) - 1
            except Exception as e:
                print(f"  - {csv_file}: Error reading file ({str(e)})")

def main():
    parser = argparse.ArgumentParser(description="Batch process ACL papers for reference extraction")
    parser.add_argument("--downloads_path", default="Downloads", help="Path to Downloads folder containing ACL folders")
    parser.add_argument("--references_path", default="References", help="Path to output References folder")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda/cpu)")
    parser.add_argument("--single_pdf", help="Process a single PDF file (for testing)")
    parser.add_argument("--extractor", choices=["layout", "text"], default="layout", help="Reference extractor: layout (pdfplumber) or text (fallback)")
    parser.add_argument("--debug", action="store_true", help="Debug folder structure without processing")

    args = parser.parse_args()
    if args.single_pdf:
        if not os.path.exists(args.single_pdf):
            print(f"PDF file not found: {args.single_pdf}")
            return

        print("Loading DeepSeek model...")
        tokenizer, model = load_model(device=args.device)

        pdf_basename = os.path.splitext(os.path.basename(args.single_pdf))[0]
        output_file = f"{pdf_basename}_references.csv"

        selected_refs = process_single_pdf(args.single_pdf, model, tokenizer, args.device, extractor=args.extractor)

        if selected_refs is not None:
            save_to_csv(selected_refs, output_file)
            print(f"Successfully processed: {args.single_pdf}")
            print(f"Results saved to: {output_file}")
        else:
            print(f"Failed to process: {args.single_pdf}")
    else:
        batch_process_acl_papers(args.downloads_path, args.references_path, args.device, extractor=args.extractor)

if __name__ == "__main__":
    main()
