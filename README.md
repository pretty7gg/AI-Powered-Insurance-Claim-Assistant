# ClaimSense AI 🏥

**AI-Powered Hospitalization Insurance Claim Assessment Platform**

An end-to-end system that automates hospitalization insurance claim review using policy-aware RAG, medical document intelligence, and LLM-driven validation — turning a manual, days-long adjudication process into an instant, evidence-backed preliminary assessment.

---

## 🧠 About

ClaimSense AI is a platform that combines **policy-aware RAG**, **medical document analysis**, **claim validation**, **evidence consistency checks**, **risk & confidence scoring**, **conversational claim assistance**, and **downloadable assessment reporting** — all built to bring transparency and speed to hospitalization insurance claims processing in India.

**How it works:**
1. 📄 Upload the insurance policy (coverage, exclusions) — chat with it to clarify coverage questions.
2. 📝 Fill out the claim form and upload supporting documents (medical bill, discharge summary, prescriptions, reports).
3. 🤖 The system extracts claim data, cross-checks it against the policy and the medical bill, and scores risk & confidence.
4. 💬 Chat with the claim assistant to understand *why* a claim passed, failed, or needs review.
5. 📥 Download a complete PDF assessment report.

---

## ✨ Key Features

- **Policy-Aware RAG Chatbot** — Ask natural-language questions about coverage/exclusions, answered strictly from the uploaded policy (FAISS + HuggingFace embeddings).
- **Automated Claim Data Extraction** — LLM (Llama 3.1 via Ollama) pulls patient, diagnosis, hospital, and bill amount from uploaded PDFs.
- **Multi-Layer Claim Validation**
  - Mandatory document check 
  - Policy exclusion detection
  - Claim form ↔ Medical Bill field-level consistency (name, diagnosis, date, hospital, amount)
- **Evidence-Based Risk & Confidence Scoring** — Transparent, weighted scoring (not a black-box fraud score) with human-readable reasoning.
- **Conversational Claim Assistant** — Ask the assistant to explain any rejection or risk flag.
- **Downloadable PDF Report** — Auto-generated report for final claim assessment.

---

## 📸 Screenshots

> _Screenshots to be added._

| Policy Upload & Chat | Claim Submission Form | Claim Assessment Result |
|---|---|---|
| _coming soon_ | _coming soon_ | _coming soon_ |

| Claim Chatbot | Downloaded PDF Report |
|---|---|
| _coming soon_ | _coming soon_ |

---

## 📥 Key Inputs

This platform relies on three categories of input to generate a claim assessment:

- **Medical Insurance Company's Handbook & Policy Documents** — coverage details, and exclusion clauses etc.
- **Claimant (Policy Holder) Details** — Personal information, medical records, and bills (e.g., medical bill, discharge summary, prescriptions, medical reports).

---

## 🏗️ Architecture

```
                         ┌──────────────────────────┐
                         │           USER           │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │       FLASK WEB UI       │
                         │  Policy Chat · Claim Form│
                         │  Document Upload · Result│
                         │  Claim Chat · PDF Report │
                         └────────────┬─────────────┘
                                      │
                ┌─────────────────────┴─────────────────────┐
                ▼                                           ▼
      ┌──────────────────────--┐                   ┌──────────────────────-┐
      │  POLICY KNOWLEDGE      │                   │   CLAIM PROCESSING    │
      │      PIPELINE          │                   │      PIPELINE         │
      ├──────────────────────  ┤                   ├────────────────────── ┤
      │ Policy PDFs            │                   │ Claim Form            │
      │(Coverage, Exclusions)  │                   │ Medical Bill,         │
      │        │               │                   │ Discharge Summary,    │
      │        ▼               │                   │ Reports, etc.         │
      │ PDF Extraction +       │                   │        │              │
      │ Text Chunking          │                   │        ▼              │
      │        │               │                   │ PDF Extraction +      │
      │        ▼               │                   │ LLM Extraction        │
      │ HuggingFace Embeddings │                   │ (Ollama · Llama 3.1)  │
      │        │               │                   │                       │
      │        ▼               │                   │                       │
      │ FAISS Vector Store     │                   │                       │
      │        │               │                   │                       │
      │        ▼               │                   │                       │
      │ Policy RAG Retrieval   │                   │                       │
      └───────────┬────────────┘                   └───────────┬───────────┘
                  │                                            │
                  └──────────────────────┬─────────────────────┘
                                          ▼
                        ┌──────────────────────────────┐
                        │    CLAIM VALIDATION ENGINE   │
                        │ • Policy Coverage/Exclusions │
                        │ • Mandatory Documents        │
                        │ • Amount Consistency         │
                        │ • Claim ↔ Bill Consistency   │      
                        └──────────────┬───────────────┘
                                       ▼
                        ┌──────────────────────────────┐
                        │       ASSESSMENT ENGINE      │
                        │ • Claim Status               │
                        │ • Risk Score & Level         │
                        │ • Confidence Score           │
                        │ • Evidence Quality           │
                        │ • AI-Generated Explanation   │
                        └──────────────┬───────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              ▼                        ▼                        ▼
       ┌─────────────┐         ┌──────────────┐         ┌───────────────┐
       │   Claim     │         │   Claim Q&A  │          │   PDF         │
       │  Assessment │         │   Chatbot    │         │   Assessment  │
       │   Result    │         │              │         │   Report      │
       └─────────────┘         └──────────────┘         └───────────────┘
```


---


---

## 📑 Sample Claim Assessment Report

Every submitted claim can be exported as a downloadable PDF report containing the overall claim status, risk score, and field-level consistency checks against the medical bill.

> _Sample report to be attached here._
