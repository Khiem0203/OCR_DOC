# OCR Vietnamese Document Full Page

Backend for processing Vietnamese documents, built on top of [MinerU](https://github.com/opendatalab/MinerU) — an open-source document OCR/parsing library by OpenDataLab.

## Introduction

This project provides a standalone backend for extracting content from Vietnamese PDF documents (administrative papers, contracts, theses, etc.), using MinerU's **VLM Auto-Engine** mode — a Vision Language Model that analyzes full document pages rather than relying on traditional OCR pipelines.

## Features

- Extract text from PDFs into Markdown and structured JSON
- Recognize mathematical formulas and tables
- Pre-processing before parsing: contrast enhancement, denoising, deskewing
- Post-processing: correct Vietnamese-specific OCR errors using rule-based and mBART model
- Supports multiple inference engines depending on the environment: vLLM (Linux), LMDeploy (Windows/Linux), MLX (macOS), Transformers (fallback)
- Async job processing — submit a file, get a `job_id`, poll for results when ready

### This project is using for R&D purpose, not for commercial.