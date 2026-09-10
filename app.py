import os
import json
import re

from flask import Flask, render_template, request, send_file
from io import BytesIO
from xml.sax.saxutils import escape
from werkzeug.utils import secure_filename

from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


app = Flask(__name__)


# =========================================================
# FOLDERS
# =========================================================

DOCUMENTS_FOLDER = "documents"
UPLOADS_FOLDER = "uploads"
FAISS_FOLDER = "faiss"

os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)
os.makedirs(UPLOADS_FOLDER, exist_ok=True)
os.makedirs(FAISS_FOLDER, exist_ok=True)


# =========================================================
# OLLAMA
# =========================================================

llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0
)

embeddings = HuggingFaceEmbeddings()


# =========================================================
# GLOBAL VARIABLES
# =========================================================

vectorstore = None

current_claim = {}

policy_chat_history = []

claim_chat_history = []


# =========================================================
# POLICY FUNCTIONS
# =========================================================

def load_policy_documents():

    documents = []

    for filename in os.listdir(DOCUMENTS_FOLDER):

        if filename.lower().endswith(".pdf"):

            file_path = os.path.join(
                DOCUMENTS_FOLDER,
                filename
            )

            loader = PyPDFLoader(file_path)

            loaded_documents = loader.load()

            documents.extend(loaded_documents)

    return documents


def split_policy_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    return chunks


def rebuild_policy_database():

    global vectorstore

    documents = load_policy_documents()

    if not documents:

        vectorstore = None
        return

    chunks = split_policy_documents(
        documents
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    vectorstore.save_local(
        FAISS_FOLDER
    )


def load_existing_policy_database():

    global vectorstore

    try:

        vectorstore = FAISS.load_local(
            FAISS_FOLDER,
            embeddings,
            allow_dangerous_deserialization=True
        )

    except Exception:

        vectorstore = None


def search_policy(question):

    if vectorstore is None:

        return "No insurance policy has been uploaded yet."

    documents = vectorstore.similarity_search(
        question,
        k=4
    )

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    return context


def answer_policy_question(question):

    conversation = ""

    for item in policy_chat_history:

        conversation += f"""
User: {item['question']}
Assistant: {item['answer']}
"""

    context = search_policy(
        question + "\n" + conversation
    )

    prompt = f"""
You are an AI insurance policy assistant.

Answer the user's question using ONLY the insurance policy
information provided below.

If the answer cannot be found in the policy, clearly say that
the information is not available in the uploaded policy.

Do not invent policy rules.

Previous conversation:
{conversation}

Policy information:
{context}

Current question:
{question}

Give a clear and simple answer.
"""

    response = llm.invoke(prompt)

    return response.content


# =========================================================
# POLICY UPLOAD
# =========================================================

def clear_old_policy_documents():

    for filename in os.listdir(
        DOCUMENTS_FOLDER
    ):

        file_path = os.path.join(
            DOCUMENTS_FOLDER,
            filename
        )

        if os.path.isfile(file_path):

            os.remove(file_path)


def clear_old_faiss_database():

    if not os.path.exists(
        FAISS_FOLDER
    ):

        return

    for filename in os.listdir(
        FAISS_FOLDER
    ):

        file_path = os.path.join(
            FAISS_FOLDER,
            filename
        )

        if os.path.isfile(file_path):

            os.remove(file_path)


def save_policy_documents(files):

    clear_old_policy_documents()

    clear_old_faiss_database()

    for file in files:

        if (
            file
            and file.filename.lower().endswith(".pdf")
        ):

            filename = secure_filename(
                file.filename
            )

            file_path = os.path.join(
                DOCUMENTS_FOLDER,
                filename
            )

            file.save(file_path)

    rebuild_policy_database()

    policy_chat_history.clear()


# =========================================================
# CLAIM DOCUMENT FUNCTIONS
# =========================================================

def save_claim_documents(files):

    saved_files = []

    for file in files:

        if (
            file
            and file.filename.lower().endswith(".pdf")
        ):

            filename = secure_filename(
                file.filename
            )

            file_path = os.path.join(
                UPLOADS_FOLDER,
                filename
            )

            file.save(file_path)

            saved_files.append(
                filename
            )

    return saved_files


def extract_pdf_text(file_path):

    loader = PyPDFLoader(
        file_path
    )

    documents = loader.load()

    text = "\n".join(
        document.page_content
        for document in documents
    )

    return text


def identify_document_type(filename):

    filename = filename.lower()

    if "bill" in filename:
        return "Medical Bill"

    if "discharge" in filename:
        return "Discharge Summary"

    if "prescription" in filename:
        return "Prescription"

    if "report" in filename:
        return "Medical Report"

    if "invoice" in filename:
        return "Invoice"

    return "Other"


# =========================================================
# NUMERIC AMOUNT EXTRACTION
# =========================================================

def extract_numeric_amount(value):
    """
    Extract numeric amount.

    Currency symbols/names are ignored.

    Examples:

    50000       -> 50000.0
    ₹50,000     -> 50000.0
    Rs. 50,000  -> 50000.0
    INR 50,000  -> 50000.0
    50,000 INR  -> 50000.0
    """

    if value is None:

        return None

    if isinstance(
        value,
        (int, float)
    ):

        return float(value)

    value = str(value).strip()

    if not value:

        return None

    cleaned = re.sub(
        r"[^\d.,\-]",
        "",
        value
    )

    if not cleaned:

        return None

    cleaned = cleaned.replace(
        ",",
        ""
    )

    try:

        return float(
            cleaned
        )

    except (ValueError, TypeError):

        return None


# =========================================================
# AI CLAIM INFORMATION EXTRACTION
# =========================================================

def extract_claim_information(files):

    combined_text = ""

    for file in files:

        file_path = os.path.join(
            UPLOADS_FOLDER,
            secure_filename(
                file.filename
            )
        )

        text = extract_pdf_text(
            file_path
        )

        combined_text += f"""

DOCUMENT: {file.filename}

{text}

-------------------------
"""

    prompt = f"""
You are an insurance claim document extraction assistant.

This application is specifically designed for
HOSPITALIZATION INSURANCE CLAIMS IN INDIA.

Extract information from the uploaded documents.

Return ONLY valid JSON. Do not include any explanation,
preamble, or text before or after the JSON object.

Use this exact structure:

{{
    "patient_name": "",
    "disease": "",
    "treatment": "",
    "hospital": "",
    "bill_amount": null,
    "expense_type": ""
}}

Important instructions:

1. bill_amount must be the FINAL medical bill amount
   if available.

2. bill_amount must contain ONLY the numeric value.

3. Ignore currency symbols and currency names.

4. Examples:

   ₹50,000 -> 50000
   Rs. 50,000 -> 50000
   INR 50,000 -> 50000
   50,000 INR -> 50000
   50000 -> 50000

5. Do NOT perform currency conversion.

6. Currency must NOT affect the extracted numeric amount.

7. Do NOT reject an amount because a currency symbol
   or currency name is present or absent.

8. Do not confuse patient age, dates, phone numbers,
   identification numbers, room numbers or other numbers
   with the final bill amount.

9. Prefer the final payable / final total bill amount.

10. If the final bill amount cannot be identified,
    return null for bill_amount.

Uploaded documents:

{combined_text}
"""

    try:

        response = llm.invoke(
            prompt
        )

        response_text = (
            response.content
            .strip()
        )

        # FIX #1: Don't assume the model's output is pure JSON.
        # Even with "return ONLY JSON" instructions, local models
        # (especially smaller ones like llama3.1:8b) frequently
        # add commentary before/after the JSON object. Previously
        # this silently failed json.loads() and reset every field
        # (including bill_amount) to None/"" on every extraction
        # that had any stray text - which is why amount checks
        # kept coming back as UNKNOWN.
        response_text = re.sub(
            r"```json\s*|\s*```",
            "",
            response_text,
            flags=re.IGNORECASE
        ).strip()

        json_match = re.search(
            r"\{.*\}",
            response_text,
            flags=re.DOTALL
        )

        if not json_match:

            print(
                "Claim extraction: no JSON object found "
                "in model output. Raw output was:"
            )
            print(response_text)

            raise ValueError(
                "No JSON object found in model output"
            )

        information = json.loads(
            json_match.group(0)
        )

        information["bill_amount"] = (
            extract_numeric_amount(
                information.get(
                    "bill_amount"
                )
            )
        )

        return {
            "patient_name":
                information.get(
                    "patient_name",
                    ""
                ),

            "disease":
                information.get(
                    "disease",
                    ""
                ),

            "treatment":
                information.get(
                    "treatment",
                    ""
                ),

            "hospital":
                information.get(
                    "hospital",
                    ""
                ),

            "bill_amount":
                information.get(
                    "bill_amount"
                ),

            "expense_type":
                information.get(
                    "expense_type",
                    ""
                )
        }

    except Exception as e:

        print(
            "Claim extraction error:",
            e
        )

        return {
            "patient_name": "",
            "disease": "",
            "treatment": "",
            "hospital": "",
            "bill_amount": None,
            "expense_type": ""
        }


# =========================================================
# CLAIM AMOUNT VALIDATION
# =========================================================

# =========================================================
# POLICY EXCLUSION CHECK
# =========================================================

def check_policy_exclusion(
    claim_information
):

    disease = claim_information.get(
        "disease",
        ""
    ).strip()

    treatment = claim_information.get(
        "treatment",
        ""
    ).strip()

    expense_type = claim_information.get(
        "expense_type",
        ""
    ).strip()

    question = f"""
Find the insurance policy rules related to exclusions
for the following hospitalization claim.

Disease:
{disease}

Treatment:
{treatment}

Expense Type:
{expense_type}

Focus specifically on:

- Excluded diseases
- Excluded treatments
- Excluded medical procedures
- Excluded hospitalization expenses
- Non-covered conditions
- Permanent exclusions
- Specific exclusions
"""

    policy_context = search_policy(
        question
    )

    if (
        not policy_context
        or
        policy_context.strip()
        == "No insurance policy has been uploaded yet."
    ):

        return {
            "status": "PASS",

            "message": (
                "No applicable policy exclusion "
                "was identified."
            )
        }

    prompt = f"""
You are an insurance policy rule evaluator.

Use ONLY the policy information provided below.

CLAIM INFORMATION:

Disease:
{disease}

Treatment:
{treatment}

Expense Type:
{expense_type}

POLICY INFORMATION:

{policy_context}

Return FAIL ONLY when the policy explicitly states
that the claim's disease, treatment, procedure,
condition or expense is excluded or not covered.

Return PASS when:

- The policy does not list the condition as excluded, OR
- No applicable exclusion is identified.

Do NOT invent an exclusion.

Do NOT return UNKNOWN.

Return ONLY:

PASS

or

FAIL
"""

    try:

        response = llm.invoke(
            prompt
        )

        result = (
            response.content
            .strip()
            .upper()
        )

        result = re.sub(
            r"[^A-Z]",
            "",
            result
        )

        if result == "FAIL":

            return {
                "status": "FAIL",

                "message": (
                    "The uploaded policy identifies "
                    "an applicable exclusion for "
                    "this hospitalization."
                )
            }

        return {
            "status": "PASS",

            "message": (
                "No applicable policy exclusion "
                "was identified for this hospitalization."
            )
        }

    except Exception as e:

        print(
            "Policy exclusion check error:",
            e
        )

        return {
            "status": "PASS",

            "message": (
                "No applicable policy exclusion "
                "was identified."
            )
        }


# =========================================================
# DOCUMENT VERIFICATION
# =========================================================

def verify_documents(
    uploaded_files
):

    # -----------------------------------------------------
    # HARD-CODED MANDATORY DOCUMENTS
    # -----------------------------------------------------

    required_documents = [
        "Medical Bill",
        "Discharge Summary"
    ]

    uploaded_types = []

    for filename in uploaded_files:

        document_type = identify_document_type(
            filename
        )

        uploaded_types.append(
            document_type
        )

    missing_documents = []

    for document in required_documents:

        if document not in uploaded_types:

            missing_documents.append(
                document
            )

    if missing_documents:

        return {
            "status": "FAIL",

            "message": (
                "Required hospitalization "
                "documents are missing."
            ),

            "missing":
                missing_documents
        }

    return {
        "status": "PASS",

        "message": (
            "Required hospitalization "
            "documents are present."
        ),

        "missing": []
    }


# =========================================================
# DOCUMENT CONSISTENCY ANALYSIS
# =========================================================

def analyze_document_consistency(
    uploaded_files
):
    """
    Uses Llama to analyze consistency across
    the uploaded claim documents .

    Returns a score from 0.0 to 1.0.
    """

    if not uploaded_files:

        return {
            "score": 0.0,

            "reason":
                "No claim documents were provided."
        }

    document_texts = []

    for filename in uploaded_files:

        try:

            file_path = os.path.join(
                UPLOADS_FOLDER,
                secure_filename(
                    filename
                )
            )

            text = extract_pdf_text(
                file_path
            )

            if text:

                document_texts.append(
                    f"""
DOCUMENT: {filename}

{text[:8000]}
"""
                )

        except Exception as e:

            print(
                f"Could not read {filename}:",
                e
            )

    if not document_texts:

        return {
            "score": 0.5,

            "reason":
                "Document text could not be reliably extracted."
        }

    combined_documents = "\n\n".join(
        document_texts
    )

    prompt = f"""
You are an insurance claim document consistency
analysis assistant.

Analyze the hospitalization claim documents below.

Determine whether the documents are internally
consistent with each other.

Check:

1. Patient name consistency
2. Diagnosis/disease consistency
3. Treatment/procedure consistency
4. Hospital consistency
5. Medical bill consistency
6. Prescription consistency
7. Discharge summary consistency
8. Contradictions between documents
9. Date consistency irrespective of time

Do NOT determine whether the claim is fraudulent.

Do NOT make a medical diagnosis.

Return ONLY valid JSON:

{{
    "consistency_score": 0.0,
    "reason": ""
}}

Rules:

- consistency_score must be between 0 and 1.
- 1.0 means highly consistent.
- 0.5 means partially consistent or insufficient information.
- 0.0 means strongly inconsistent.
- Do not invent information.
- Explain the main reason briefly.

DOCUMENTS:

{combined_documents}
"""

    try:

        response = llm.invoke(
            prompt
        )

        content = response.content.strip()

        content = re.sub(
            r"```json\s*|\s*```",
            "",
            content,
            flags=re.IGNORECASE
        ).strip()

        json_match = re.search(
            r"\{.*\}",
            content,
            flags=re.DOTALL
        )

        if not json_match:

            print(
                "Document consistency: no JSON object found "
                "in model output. Raw output was:"
            )
            print(content)

            raise ValueError(
                "No JSON object found in model output"
            )

        result = json.loads(
            json_match.group(0)
        )

        score = float(
            result.get(
                "consistency_score",
                0.5
            )
        )

        score = max(
            0.0,
            min(1.0, score)
        )

        return {
            "score": score,

            "reason":
                result.get(
                    "reason",
                    "Documents were analyzed for consistency."
                )
        }

    except Exception as e:

        print(
            "Document consistency analysis error:",
            e
        )

        return {
            "score": 0.5,

            "reason":
                "Document consistency could not be reliably determined."
        }



# =========================================================
# CLAIM FORM VS MEDICAL BILL CONSISTENCY
# =========================================================

def check_claim_form_vs_medical_bill(
    claim_data,
    uploaded_files
):
    """
    Compare the claim-form fields against the submitted Medical Bill.

    Checked:
        1. Patient Name
        2. Hospitalization Reason / Diagnosis
        3. Hospitalization Date
        4. Hospital / Medical Facility
        5. Claim Amount

    Claim amount comparison is handled here so that all
    claim-form vs Medical Bill consistency checks are returned
    together in one result.
    """

    from difflib import SequenceMatcher
    from datetime import datetime

    # -----------------------------------------------------
    # HELPER: NORMALIZE TEXT
    # -----------------------------------------------------

    def normalize_text(value):
        if value is None:
            return ""

        value = str(value).lower().strip()

        # Remove punctuation
        value = re.sub(r"[^a-z0-9\s]", " ", value)

        # Normalize whitespace
        value = re.sub(r"\s+", " ", value)

        return value.strip()

    # -----------------------------------------------------
    # HELPER: COMPARE NAMES / HOSPITALS
    # -----------------------------------------------------

    def text_matches(submitted, bill_value):

        submitted_norm = normalize_text(submitted)
        bill_norm = normalize_text(bill_value)

        if not submitted_norm or not bill_norm:
            return False

        # Exact match
        if submitted_norm == bill_norm:
            return True

        # One value contained in the other
        if (
            submitted_norm in bill_norm
            or bill_norm in submitted_norm
        ):
            return True

        # Token comparison
        submitted_tokens = set(
            submitted_norm.split()
        )

        bill_tokens = set(
            bill_norm.split()
        )

        if submitted_tokens and bill_tokens:

            common_tokens = (
                submitted_tokens &
                bill_tokens
            )

            # Require all submitted tokens to
            # appear in the bill where possible.
            if (
                len(common_tokens)
                == len(submitted_tokens)
            ):
                return True

        # Fuzzy comparison
        similarity = SequenceMatcher(
            None,
            submitted_norm,
            bill_norm
        ).ratio()

        return similarity >= 0.88

    # -----------------------------------------------------
    # HELPER: DATE NORMALIZATION
    # -----------------------------------------------------

    def normalize_date(value):

        if not value:
            return None

        value = str(value).strip()

        formats = [
            "%Y-%m-%d",
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%d.%m.%Y",
            "%Y/%m/%d",
            "%d-%b-%Y",
            "%d-%B-%Y",
            "%d %b %Y",
            "%d %B %Y"
        ]

        for fmt in formats:

            try:

                return datetime.strptime(
                    value,
                    fmt
                ).date()

            except ValueError:
                pass

        return None

    # -----------------------------------------------------
    # FIND MEDICAL BILL
    # -----------------------------------------------------

    medical_bill_filename = None

    for filename in uploaded_files:

        if (
            identify_document_type(filename)
            == "Medical Bill"
        ):
            medical_bill_filename = filename
            break

    if not medical_bill_filename:

        return {
            "status": "REVIEW",
            "message": (
                "Medical Bill was not found, so "
                "claim-form consistency could not "
                "be checked."
            ),
            "checks": []
        }

    # -----------------------------------------------------
    # READ MEDICAL BILL
    # -----------------------------------------------------

    bill_path = os.path.join(
        UPLOADS_FOLDER,
        secure_filename(
            medical_bill_filename
        )
    )

    try:

        bill_text = extract_pdf_text(
            bill_path
        )

    except Exception as e:

        print(
            "Medical Bill consistency extraction error:",
            e
        )

        return {
            "status": "REVIEW",
            "message": (
                "Medical Bill text could not be "
                "reliably extracted."
            ),
            "checks": []
        }

    if not bill_text or not bill_text.strip():

        return {
            "status": "REVIEW",
            "message": (
                "Medical Bill contains no readable "
                "text, so consistency could not "
                "be checked."
            ),
            "checks": []
        }

    # -----------------------------------------------------
    # USER SUBMITTED VALUES
    # -----------------------------------------------------

    submitted_patient_name = (
        claim_data.get(
            "patient_name",
            ""
        )
        or ""
    ).strip()

    submitted_diagnosis = (
        claim_data.get(
            "hospitalization_reason",
            ""
        )
        or ""
    ).strip()

    submitted_date = (
        claim_data.get(
            "claim_date",
            ""
        )
        or ""
    ).strip()

    submitted_facility = (
        claim_data.get(
            "medical_facility",
            ""
        )
        or ""
    ).strip()

    submitted_claim_amount = extract_numeric_amount(
        claim_data.get(
            "claim_amount"
        )
    )

    # -----------------------------------------------------
    # FIRST STEP:
    # ASK LLAMA TO EXTRACT ONLY WHAT IS ACTUALLY
    # WRITTEN IN THE MEDICAL BILL
    # -----------------------------------------------------

    extraction_prompt = f"""
You are extracting factual information from an
insurance Medical Bill.

IMPORTANT:
Use ONLY information explicitly present in the
Medical Bill text.

Do NOT guess.
Do NOT infer.
Do NOT use the submitted claim form values.
Do NOT assume that two values match.

Extract these five fields:

1. Patient Name
2. Hospitalization Reason / Diagnosis
3. Hospitalization Date
4. Hospital / Medical Facility
5. Final Medical Bill Amount

For every field provide:
- value
- evidence

The evidence MUST be an exact short phrase copied
from the Medical Bill.

For Final Medical Bill Amount:
- Extract the final payable / final total bill amount.
- Return ONLY the numeric amount in value.
- Ignore currency symbols/names.
- Do not use patient age, dates, phone numbers, IDs, room numbers,
  item-level charges, subtotal values, or other unrelated numbers.
- If the final bill amount cannot be reliably identified, return null.

If a field is not present, return an empty value
and empty evidence.

Return ONLY valid JSON:

{{
    "patient_name": {{
        "value": "",
        "evidence": ""
    }},
    "diagnosis": {{
        "value": "",
        "evidence": ""
    }},
    "hospitalization_date": {{
        "value": "",
        "evidence": ""
    }},
    "medical_facility": {{
        "value": "",
        "evidence": ""
    }},
    "bill_amount": {{
        "value": null,
        "evidence": ""
    }}
}}

MEDICAL BILL:
{bill_text[:12000]}
"""

    try:

        response = llm.invoke(
            extraction_prompt
        )

        content = response.content.strip()

        content = re.sub(
            r"```json\s*|\s*```",
            "",
            content,
            flags=re.IGNORECASE
        ).strip()

        json_match = re.search(
            r"\{.*\}",
            content,
            flags=re.DOTALL
        )

        if not json_match:
            raise ValueError(
                "No JSON object found in extraction response"
            )

        extracted = json.loads(
            json_match.group(0)
        )

    except Exception as e:

        print(
            "Medical Bill field extraction error:",
            e
        )

        return {
            "status": "REVIEW",
            "message": (
                "The Medical Bill fields could not "
                "be reliably extracted."
            ),
            "checks": []
        }

    # -----------------------------------------------------
    # EXTRACTED BILL VALUES
    # -----------------------------------------------------

    bill_patient = (
        extracted.get(
            "patient_name",
            {}
        ).get("value", "")
        or ""
    ).strip()

    bill_patient_evidence = (
        extracted.get(
            "patient_name",
            {}
        ).get("evidence", "")
        or ""
    ).strip()

    bill_diagnosis = (
        extracted.get(
            "diagnosis",
            {}
        ).get("value", "")
        or ""
    ).strip()

    bill_diagnosis_evidence = (
        extracted.get(
            "diagnosis",
            {}
        ).get("evidence", "")
        or ""
    ).strip()

    bill_date = (
        extracted.get(
            "hospitalization_date",
            {}
        ).get("value", "")
        or ""
    ).strip()

    bill_date_evidence = (
        extracted.get(
            "hospitalization_date",
            {}
        ).get("evidence", "")
        or ""
    ).strip()

    bill_facility = (
        extracted.get(
            "medical_facility",
            {}
        ).get("value", "")
        or ""
    ).strip()

    bill_facility_evidence = (
        extracted.get(
            "medical_facility",
            {}
        ).get("evidence", "")
        or ""
    ).strip()

    bill_amount = extract_numeric_amount(
        extracted.get(
            "bill_amount",
            {}
        ).get("value")
    )

    bill_amount_evidence = (
        extracted.get(
            "bill_amount",
            {}
        ).get("evidence", "")
        or ""
    ).strip()

    # -----------------------------------------------------
    # VERIFY THAT LLAMA'S EXTRACTED VALUES ACTUALLY
    # APPEAR IN THE MEDICAL BILL
    # -----------------------------------------------------

    bill_text_normalized = normalize_text(
        bill_text
    )

    def evidence_exists(evidence):

        if not evidence:
            return False

        evidence_norm = normalize_text(
            evidence
        )

        if not evidence_norm:
            return False

        return (
            evidence_norm
            in bill_text_normalized
        )

    # -----------------------------------------------------
    # 1. PATIENT NAME
    # -----------------------------------------------------

    if not bill_patient:

        patient_status = "REVIEW"

        patient_reason = (
            "Patient name could not be identified "
            "in the Medical Bill."
        )

    elif not evidence_exists(
        bill_patient_evidence
    ):

        patient_status = "REVIEW"

        patient_reason = (
            "The extracted patient name could not "
            "be verified against the Medical Bill text."
        )

    elif not submitted_patient_name:

        patient_status = "REVIEW"

        patient_reason = (
            "Patient name was not entered in the "
            "claim form."
        )

    elif text_matches(
        submitted_patient_name,
        bill_patient
    ):

        patient_status = "PASS"

        patient_reason = (
            "The submitted patient name matches "
            "the Medical Bill."
        )

    else:

        patient_status = "FAIL"

        patient_reason = (
            "The submitted patient name does not "
            "match the Medical Bill."
        )

    # -----------------------------------------------------
    # 2. HOSPITAL / MEDICAL FACILITY
    # -----------------------------------------------------

    if not bill_facility:

        facility_status = "REVIEW"

        facility_reason = (
            "Hospital / Medical Facility could not "
            "be identified in the Medical Bill."
        )

    elif not evidence_exists(
        bill_facility_evidence
    ):

        facility_status = "REVIEW"

        facility_reason = (
            "The extracted hospital name could not "
            "be verified against the Medical Bill text."
        )

    elif not submitted_facility:

        facility_status = "REVIEW"

        facility_reason = (
            "Hospital / Medical Facility was not "
            "entered in the claim form."
        )

    elif text_matches(
        submitted_facility,
        bill_facility
    ):

        facility_status = "PASS"

        facility_reason = (
            "The submitted hospital / facility "
            "matches the Medical Bill."
        )

    else:

        facility_status = "FAIL"

        facility_reason = (
            "The submitted hospital / facility "
            "does not match the Medical Bill."
        )

    # -----------------------------------------------------
    # 3. HOSPITALIZATION DATE
    # -----------------------------------------------------

    submitted_date_obj = normalize_date(
        submitted_date
    )

    bill_date_obj = normalize_date(
        bill_date
    )

    if not bill_date:

        date_status = "REVIEW"

        date_reason = (
            "Hospitalization date could not be "
            "identified in the Medical Bill."
        )

    elif not evidence_exists(
        bill_date_evidence
    ):

        date_status = "REVIEW"

        date_reason = (
            "The extracted hospitalization date "
            "could not be verified against the "
            "Medical Bill text."
        )

    elif not submitted_date:

        date_status = "REVIEW"

        date_reason = (
            "Hospitalization date was not entered "
            "in the claim form."
        )

    elif (
        submitted_date_obj is not None
        and bill_date_obj is not None
    ):

        if submitted_date_obj == bill_date_obj:

            date_status = "PASS"

            date_reason = (
                "The hospitalization date matches "
                "the Medical Bill."
            )

        else:

            date_status = "FAIL"

            date_reason = (
                "The submitted hospitalization date "
                "does not match the Medical Bill."
            )

    else:

        # If dates could not be parsed reliably,
        # do NOT let the LLM decide that they match.

        submitted_date_norm = normalize_text(
            submitted_date
        )

        bill_date_norm = normalize_text(
            bill_date
        )

        if (
            submitted_date_norm
            == bill_date_norm
        ):

            date_status = "PASS"

            date_reason = (
                "The hospitalization date matches "
                "the Medical Bill."
            )

        else:

            date_status = "FAIL"

            date_reason = (
                "The submitted hospitalization date "
                "does not match the Medical Bill."
            )

    # -----------------------------------------------------
    # 4. DIAGNOSIS
    # -----------------------------------------------------
    #
    # Diagnosis is semantic, so use Llama here.
    # BUT require evidence from the Medical Bill.
    # -----------------------------------------------------

    if not bill_diagnosis:

        diagnosis_status = "REVIEW"

        diagnosis_reason = (
            "Diagnosis could not be identified in "
            "the Medical Bill."
        )

    elif not evidence_exists(
        bill_diagnosis_evidence
    ):

        diagnosis_status = "REVIEW"

        diagnosis_reason = (
            "The extracted diagnosis could not "
            "be verified against the Medical Bill."
        )

    elif not submitted_diagnosis:

        diagnosis_status = "REVIEW"

        diagnosis_reason = (
            "Hospitalization Reason / Diagnosis "
            "was not entered in the claim form."
        )

    else:

        diagnosis_prompt = f"""
You are comparing TWO pieces of information.

SUBMITTED CLAIM FORM DIAGNOSIS:
{submitted_diagnosis}

MEDICAL BILL DIAGNOSIS:
{bill_diagnosis}

MEDICAL BILL EVIDENCE:
{bill_diagnosis_evidence}

Determine whether the submitted diagnosis is
consistent with the diagnosis documented in the
Medical Bill.

Rules:

PASS:
- Same diagnosis
- Clearly equivalent medical terminology
- Abbreviation of the same condition
- Different wording describing the same condition

FAIL:
- Clearly different disease
- Clearly different medical condition
- The submitted diagnosis contradicts the bill

REVIEW:
- Insufficient information
- Ambiguous
- Cannot reliably determine equivalence

Do NOT guess.
Do NOT invent a relationship.

Return ONLY one word:

PASS
FAIL
REVIEW
"""

        try:

            diagnosis_response = llm.invoke(
                diagnosis_prompt
            )

            diagnosis_status = (
                diagnosis_response.content
                .strip()
                .upper()
            )

            if diagnosis_status not in {
                "PASS",
                "FAIL",
                "REVIEW"
            }:

                diagnosis_status = "REVIEW"

            if diagnosis_status == "PASS":

                diagnosis_reason = (
                    "The submitted diagnosis is "
                    "consistent with the Medical Bill."
                )

            elif diagnosis_status == "FAIL":

                diagnosis_reason = (
                    "The submitted diagnosis conflicts "
                    "with the Medical Bill."
                )

            else:

                diagnosis_reason = (
                    "The diagnosis could not be "
                    "reliably compared."
                )

        except Exception as e:

            print(
                "Diagnosis consistency error:",
                e
            )

            diagnosis_status = "REVIEW"

            diagnosis_reason = (
                "The diagnosis could not be "
                "reliably compared."
            )

    # -----------------------------------------------------
    # 5. CLAIM AMOUNT
    # -----------------------------------------------------

    if bill_amount is None:

        amount_status = "REVIEW"

        amount_reason = (
            "The final Medical Bill amount could not be "
            "reliably identified."
        )

    elif not evidence_exists(
        bill_amount_evidence
    ):

        amount_status = "REVIEW"

        amount_reason = (
            "The extracted Medical Bill amount could not "
            "be verified against the Medical Bill text."
        )

    elif submitted_claim_amount is None:

        amount_status = "REVIEW"

        amount_reason = (
            "Claim amount was not entered in the claim form."
        )

    else:

        if submitted_claim_amount <= bill_amount:

            amount_status = "PASS"

            amount_reason = (
                "The submitted claim amount is within the "
                "final Medical Bill amount."
            )

        else:

            amount_status = "FAIL"

            amount_reason = (
                f"Submitted claim amount of "
                f"₹{submitted_claim_amount:,.2f} is greater than "
                f"the Medical Bill amount of ₹{bill_amount:,.2f}."
            )

    # -----------------------------------------------------
    # BUILD CHECK RESULTS
    # -----------------------------------------------------

    checks = [

        {
            "field": "Patient Name",
            "status": patient_status,
            "submitted_value": submitted_patient_name,
            "bill_value": bill_patient,
            "reason": patient_reason
        },

        {
            "field": "Hospitalization Reason / Diagnosis",
            "status": diagnosis_status,
            "submitted_value": submitted_diagnosis,
            "bill_value": bill_diagnosis,
            "reason": diagnosis_reason
        },

        {
            "field": "Hospitalization Date",
            "status": date_status,
            "submitted_value": submitted_date,
            "bill_value": bill_date,
            "reason": date_reason
        },

        {
            "field": "Hospital / Medical Facility",
            "status": facility_status,
            "submitted_value": submitted_facility,
            "bill_value": bill_facility,
            "reason": facility_reason
        },

        {
            "field": "Claim Amount",
            "status": amount_status,
            "submitted_value": (
                f"₹{submitted_claim_amount:,.2f}"
                if submitted_claim_amount is not None
                else ""
            ),
            "bill_value": (
                f"₹{bill_amount:,.2f}"
                if bill_amount is not None
                else ""
            ),
            "reason": amount_reason
        }
    ]

    # -----------------------------------------------------
    # FINAL STATUS
    # -----------------------------------------------------

    statuses = [
        check["status"]
        for check in checks
    ]

    if "FAIL" in statuses:

        overall_status = "FAIL"

        message = (
            "One or more claim-form fields conflict "
            "with the Medical Bill."
        )

    elif all(
        status == "PASS"
        for status in statuses
    ):

        overall_status = "PASS"

        message = (
            "All claim-form fields are consistent "
            "with the Medical Bill."
        )

    else:

        overall_status = "REVIEW"

        message = (
            "The claim-form fields could not all be "
            "reliably verified against the Medical Bill."
        )

    # -----------------------------------------------------
    # RETURN RESULT
    # -----------------------------------------------------

    return {
        "status": overall_status,
        "message": message,
        "checks": checks
    }





# =========================================================
# CLAIM VALIDATION
# =========================================================

def validate_claim(
    claim_information,
    uploaded_files,
    claim_data=None
):

    # -----------------------------------------------------
    # 1. DOCUMENT CHECK
    # -----------------------------------------------------

    document_result = verify_documents(
        uploaded_files
    )

    # -----------------------------------------------------
    # 2. POLICY EXCLUSION CHECK
    # -----------------------------------------------------

    exclusion_result = check_policy_exclusion(
        claim_information
    )

    # -----------------------------------------------------
    # 3. CLAIM FORM VS MEDICAL BILL
    # -----------------------------------------------------

    claim_bill_result = check_claim_form_vs_medical_bill(
        claim_data or {},
        uploaded_files
    )

    # -----------------------------------------------------
    # 4. AMOUNT CHECK
    #
    # Amount is already checked inside
    # Claim Form vs Medical Bill.
    #
    # We extract the Claim Amount check from there
    # so the overall claim status can use it as a
    # FAIL condition.
    # -----------------------------------------------------

    amount_result = next(
        (
            check
            for check in claim_bill_result.get(
                "checks",
                []
            )
            if check.get("field") == "Claim Amount"
        ),
        {
            "field": "Claim Amount",
            "status": "REVIEW",
            "reason": (
                "Claim amount could not be reliably "
                "verified against the Medical Bill."
            )
        }
    )

    # -----------------------------------------------------
    # 5. OVERALL CLAIM STATUS
    # -----------------------------------------------------
    #
    # PRIORITY:
    #
    # Policy exclusion applicable -> FAIL
    # Amount check FAIL            -> FAIL
    # Mandatory document missing   -> REVIEW
    # Claim vs Bill inconsistency  -> REVIEW
    # Everything passes            -> PASS
    #
    # -----------------------------------------------------

    # =====================================================
    # CONDITION 1: POLICY EXCLUSION
    # =====================================================

    if exclusion_result.get("status") == "FAIL":

        overall_status = "FAIL"

    # =====================================================
    # CONDITION 2: AMOUNT CHECK
    # =====================================================

    elif amount_result.get("status") == "FAIL":

        overall_status = "FAIL"

    # =====================================================
    # CONDITION 3: MANDATORY DOCUMENTS
    # =====================================================

    elif document_result.get("status") in (
        "FAIL",
        "REVIEW"
    ):

        overall_status = "REVIEW"

    # =====================================================
    # CONDITION 4: CLAIM FORM VS MEDICAL BILL
    # =====================================================

    elif claim_bill_result.get("status") in (
        "FAIL",
        "REVIEW"
    ):

        overall_status = "REVIEW"

    # =====================================================
    # CONDITION 5: EVERYTHING PASSED
    # =====================================================

    else:

        overall_status = "PASS"

    # -----------------------------------------------------
    # RETURN RESULT
    # -----------------------------------------------------

    return {
        "overall_status":
            overall_status,

        "amount_check":
            amount_result,

        "document_check":
            document_result,

        "exclusion_check":
            exclusion_result,

        "claim_bill_consistency":
            claim_bill_result
    }


# =========================================================
# HELPER
# =========================================================

def clamp_score(value):

    return max(
        0,
        min(
            100,
            round(value)
        )
    )


# =========================================================
# CLAIM RISK SCORE
# =========================================================

def calculate_claim_risk(
    validation_results,
    claim_information,
    uploaded_files,
    document_consistency_score=1.0
):
    """
    Evidence-based Claim Risk Score.

    This is NOT a fraud probability.

    Policy exclusion is treated as a gating condition:
        - FAIL -> claim is high risk regardless of other evidence
        - PASS -> calculate risk using:
            35% Amount Consistency
            25% Document Completeness
            40% Claim Form vs Medical Bill Consistency

    Lower score  = lower risk
    Higher score = higher risk
    """

    reasons = []

    # =====================================================
    # 1. POLICY EXCLUSION - GATING CHECK
    # =====================================================

    exclusion_result = validation_results.get(
        "exclusion_check",
        {}
    )

    exclusion_status = exclusion_result.get(
        "status",
        "PASS"
    )

    policy_consistency = 100

    if exclusion_status == "FAIL":

        policy_consistency = 0

        reasons.append(
            "The uploaded insurance policy identifies "
            "an applicable exclusion for this hospitalization. "
            "Therefore, the claim is not covered under the "
            "identified policy rule."
        )

        # -------------------------------------------------
        # POLICY EXCLUSION OVERRIDES OTHER RISK SIGNALS
        # -------------------------------------------------

        risk_score = 95
        risk_level = "HIGH"

        evidence_quality = 5

        return {
            "score": risk_score,
            "level": risk_level,
            "evidence_quality": evidence_quality,
            "why": reasons,
            "signals": {
                "amount_consistency": 0,
                "document_completeness": 0,
                "claim_bill_consistency": 0,
                "policy_consistency": 0
            }
        }

    # =====================================================
    # 2. AMOUNT CONSISTENCY
    # =====================================================

    amount_result = validation_results.get(
        "amount_check",
        {}
    )

    amount_status = amount_result.get(
        "status",
        "UNKNOWN"
    )

    if amount_status == "PASS":

        amount_consistency = 100

    elif amount_status == "FAIL":

        amount_consistency = 0

        reasons.append(
            "The submitted claim amount is greater than "
            "the medical bill amount."
        )

    else:

        amount_consistency = 0

        reasons.append(
            "The claim amount or medical bill amount "
            "could not be reliably determined."
        )

    # =====================================================
    # 3. DOCUMENT COMPLETENESS
    # =====================================================

    required_documents = [
        "Medical Bill",
        "Discharge Summary"
    ]

    uploaded_document_types = [
        identify_document_type(filename)
        for filename in uploaded_files
    ]

    documents_present = sum(
        1
        for document in required_documents
        if document in uploaded_document_types
    )

    if required_documents:

        document_completeness = (
            documents_present
            /
            len(required_documents)
        ) * 100

    else:

        document_completeness = 100

    missing_documents = [
        document
        for document in required_documents
        if document not in uploaded_document_types
    ]

    if missing_documents:

        reasons.append(
            "Missing required document(s): "
            + ", ".join(missing_documents)
            + "."
        )

    # =====================================================
    # 4. CLAIM FORM VS MEDICAL BILL CONSISTENCY
    # =====================================================

    claim_bill_result = validation_results.get(
        "claim_bill_consistency",
        {}
    )

    claim_bill_status = claim_bill_result.get(
        "status",
        "REVIEW"
    )

    checks = claim_bill_result.get(
        "checks",
        []
    )

    if checks:

        check_scores = []

        for check in checks:

            status = check.get(
                "status",
                "REVIEW"
            )

            if status == "PASS":

                check_scores.append(100)

            elif status == "FAIL":

                check_scores.append(0)

            else:

                # REVIEW means evidence is insufficient,
                # not that the claim is definitely wrong.
                check_scores.append(50)

        claim_bill_consistency = (
            sum(check_scores)
            /
            len(check_scores)
        )

    elif claim_bill_status == "PASS":

        claim_bill_consistency = 100

    elif claim_bill_status == "FAIL":

        claim_bill_consistency = 0

        reasons.append(
            "One or more claim-form fields conflict "
            "with the Medical Bill."
        )

    else:

        claim_bill_consistency = 50

        reasons.append(
            "The Claim Form could not be completely "
            "verified against the Medical Bill."
        )

    # =====================================================
    # 5. POLICY PASSED
    # =====================================================

    policy_consistency = 100

    # =====================================================
    # 6. EVIDENCE QUALITY
    # =====================================================
    #
    # Policy exclusion is NOT included here because it
    # has already acted as a gating condition.
    #
    # Covered claims are evaluated using:
    #
    # Amount Consistency          = 35%
    # Document Completeness       = 25%
    # Claim Form vs Medical Bill  = 40%
    #
    # =====================================================

    evidence_quality = (
        amount_consistency * 0.35
        +
        document_completeness * 0.25
        +
        claim_bill_consistency * 0.40
    )

    evidence_quality = clamp_score(
        evidence_quality
    )

    # =====================================================
    # 7. FINAL CLAIM RISK
    # =====================================================

    risk_score = clamp_score(
        100 - evidence_quality
    )

    # =====================================================
    # 8. RISK LEVEL
    # =====================================================

    if risk_score < 25:

        risk_level = "LOW"

    elif risk_score < 50:

        risk_level = "MEDIUM"

    else:

        risk_level = "HIGH"

    # =====================================================
    # 9. POSITIVE REASON FOR LOW RISK
    # =====================================================

    if not reasons:

        reasons.append(
            "The claim evidence is consistent, required "
            "documents are present, the claim amount is "
            "supported by the medical bill, and no "
            "applicable policy exclusion was identified."
        )

    # =====================================================
    # 10. RETURN RESULT
    # =====================================================

    return {
        "score": risk_score,

        "level": risk_level,

        "evidence_quality": evidence_quality,

        "why": reasons,

        "signals": {

            "amount_consistency":
                round(amount_consistency),

            "document_completeness":
                round(document_completeness),

            "claim_bill_consistency":
                round(claim_bill_consistency),

            "policy_consistency":
                round(policy_consistency)
        }
    }


# =========================================================
# CLAIM CONFIDENCE
# =========================================================

def calculate_claim_confidence(
    validation_results,
    claim_information,
    uploaded_files,
    document_consistency_result
):
    """
    Confidence represents how strong and complete the
    available evidence is.

    IMPORTANT:

    Risk and confidence are independent.

    High Risk + High Confidence is possible.
    """

    confidence_components = []


    # -----------------------------------------------------
    # AMOUNT EVIDENCE
    # -----------------------------------------------------

    claim_bill_result = validation_results.get(
        "claim_bill_consistency",
        {}
    )

    amount_result = next(
        (
            check
            for check in claim_bill_result.get("checks", [])
            if check.get("field") == "Claim Amount"
        ),
        {}
    )

    amount_status = str(
        amount_result.get(
            "status",
            "REVIEW"
        )
    ).upper()

    if amount_status in [
        "PASS",
        "FAIL"
    ]:

        amount_confidence = 100

    else:

        amount_confidence = 20

    confidence_components.append(
        amount_confidence
    )


    # -----------------------------------------------------
    # DOCUMENT EVIDENCE
    # -----------------------------------------------------

    if uploaded_files:

        document_confidence = 100

    else:

        document_confidence = 0

    confidence_components.append(
        document_confidence
    )


    # -----------------------------------------------------
    # INFORMATION COMPLETENESS
    # -----------------------------------------------------

    important_fields = [
        "patient_name",
        "disease",
        "treatment",
        "hospital",
        "bill_amount",
        "expense_type"
    ]

    available_fields = sum(
        1
        for field in important_fields
        if claim_information.get(field)
    )

    information_confidence = (
        available_fields
        /
        len(important_fields)
    ) * 100

    confidence_components.append(
        information_confidence
    )


    # -----------------------------------------------------
    # DOCUMENT CONSISTENCY
    # -----------------------------------------------------

    consistency_score = document_consistency_result.get(
        "score",
        0.5
    )

    consistency_confidence = (
        consistency_score * 100
    )

    confidence_components.append(
        consistency_confidence
    )


    # -----------------------------------------------------
    # POLICY EVIDENCE
    # -----------------------------------------------------

    exclusion_result = validation_results.get(
        "exclusion_check",
        {}
    )

    if exclusion_result.get("status") in [
        "PASS",
        "FAIL"
    ]:

        policy_confidence = 100

    else:

        policy_confidence = 50

    confidence_components.append(
        policy_confidence
    )


    # -----------------------------------------------------
    # FINAL CONFIDENCE
    # -----------------------------------------------------

    confidence = (
        sum(confidence_components)
        /
        len(confidence_components)
    )

    return clamp_score(
        confidence
    )


# =========================================================
# AI EXPLANATION
# =========================================================

def generate_claim_explanation(
    claim_data,
    validation_result,
    risk_result,
    confidence
):

    prompt = f"""
You are an AI hospitalization insurance claim assistant.

The application is designed for hospitalization insurance
claims in India.

All monetary values are displayed as INR (₹).

Provide a simple explanation of the preliminary claim
assessment.

Claim information:

{json.dumps(
    claim_data,
    indent=2
)}

Validation:

{json.dumps(
    validation_result,
    indent=2
)}

Claim risk assessment:

{json.dumps(
    risk_result,
    indent=2
)}

Confidence:

{confidence}%

Explain:

1. What was checked
2. What passed
3. What failed or needs review
4. Why the claim received its current risk level
5. What the customer should do next

Important:

The Claim Risk Score is an evidence-based risk indicator.
It is NOT a probability of fraud.

This is an AI-assisted preliminary assessment.

It is not a final insurance approval or rejection.
Human insurer review may still be required.
"""

    response = llm.invoke(
        prompt
    )

    return response.content


# =========================================================
# CLAIM CHATBOT
# =========================================================

def answer_claim_question(
    question
):

    conversation = ""

    for item in claim_chat_history:

        conversation += f"""
User: {item['question']}
Assistant: {item['answer']}
"""

    prompt = f"""
You are a conversational hospitalization insurance
claim assistant.

The application is designed for hospitalization claims
in India.

Monetary values are displayed as INR (₹).

Current claim information:

{json.dumps(
    current_claim,
    indent=2
)}

Previous conversation:

{conversation}

User question:

{question}

Answer clearly and simply.

Use the claim information and uploaded policy
information when relevant.

Do not invent policy rules.

If the user asks whether the claim is finally approved
or rejected, explain that the system provides a
preliminary AI-assisted assessment and human insurer
review may still be required.
"""

    response = llm.invoke(
        prompt
    )

    return response.content


# =========================================================
# ROUTES
# =========================================================

@app.route("/")
def home():

    policy_files = []

    for filename in os.listdir(
        DOCUMENTS_FOLDER
    ):

        if filename.lower().endswith(
            ".pdf"
        ):

            policy_files.append(
                filename
            )

    return render_template(
        "index.html",

        policy_files=
            policy_files,

        policy_chat_history=
            policy_chat_history
    )


# =========================================================
# UPLOAD POLICY
# =========================================================

@app.route(
    "/upload-policy",
    methods=["POST"]
)
def upload_policy():

    files = request.files.getlist(
        "policy_files"
    )

    save_policy_documents(
        files
    )

    return home()


# =========================================================
# POLICY CHAT
# =========================================================

@app.route(
    "/ask-policy",
    methods=["POST"]
)
def ask_policy():

    question = request.form.get(
        "question",
        ""
    ).strip()

    if question:

        answer = answer_policy_question(
            question
        )

        policy_chat_history.append({

            "question":
                question,

            "answer":
                answer
        })

    return home()


# =========================================================
# SUBMIT HOSPITALIZATION CLAIM
# =========================================================

@app.route(
    "/submit-claim",
    methods=["POST"]
)
def submit_claim():

    global current_claim
    global claim_chat_history

    claim_chat_history = []

    patient_name = request.form.get(
        "patient_name",
        ""
    )

    address = request.form.get(
        "address",
        ""
    )

    hospitalization_reason = request.form.get(
        "hospitalization_reason",
        ""
    )

    claim_date = request.form.get(
        "claim_date",
        ""
    )

    medical_facility = request.form.get(
        "medical_facility",
        ""
    )

    claim_amount = request.form.get(
        "claim_amount",
        "0"
    )

    description = request.form.get(
        "description",
        ""
    )

    documents = request.files.getlist(
        "documents"
    )

    # -----------------------------------------------------
    # SAVE DOCUMENTS
    # -----------------------------------------------------

    saved_files = save_claim_documents(
        documents
    )

    # -----------------------------------------------------
    # EXTRACT INFORMATION
    # -----------------------------------------------------

    extracted_information = (
        extract_claim_information(
            documents
        )
    )

    # -----------------------------------------------------
    # NORMALIZE CLAIM AMOUNT
    # -----------------------------------------------------

    numeric_claim_amount = (
        extract_numeric_amount(
            claim_amount
        )
    )

    # -----------------------------------------------------
    # DEBUG LOGGING
    # (helps diagnose amount / risk assessment issues -
    # check your console output after submitting a claim)
    # -----------------------------------------------------

    print("EXTRACTED INFORMATION:", extracted_information)
    print("SUBMITTED CLAIM AMOUNT:", claim_amount, "->", numeric_claim_amount)

    # -----------------------------------------------------
    # CLAIM DATA
    # -----------------------------------------------------

    claim_data = {

        "patient_name":
            patient_name,

        "address":
            address,

        "claim_type":
            "Hospitalization",

        "hospitalization_reason":
            hospitalization_reason,

        "claim_date":
            claim_date,

        "medical_facility":
            medical_facility,

        "claim_amount":
            numeric_claim_amount,

        "currency":
            "INR",

        "description":
            description,

        "uploaded_documents":
            saved_files,

        "extracted_information":
            extracted_information
    }

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    validation_result = validate_claim(

        extracted_information,

        saved_files,

        claim_data
    )

    print("VALIDATION RESULT:", validation_result)

    # -----------------------------------------------------
    # DOCUMENT CONSISTENCY
    # -----------------------------------------------------

    document_consistency_result = (
        analyze_document_consistency(
            saved_files
        )
    )

    # -----------------------------------------------------
    # CLAIM RISK
    # -----------------------------------------------------

    risk_result = calculate_claim_risk(

    validation_result,

    extracted_information,

    saved_files
)

    # -----------------------------------------------------
    # CONFIDENCE
    # -----------------------------------------------------

    confidence = calculate_claim_confidence(

        validation_result,

        extracted_information,

        saved_files,

        document_consistency_result
    )

    # -----------------------------------------------------
    # AI EXPLANATION
    # -----------------------------------------------------

    explanation = (
        generate_claim_explanation(

            claim_data,

            validation_result,

            risk_result,

            confidence
        )
    )

    # -----------------------------------------------------
    # STORE CURRENT CLAIM
    # -----------------------------------------------------

    current_claim = {

        "claim":
            claim_data,

        "validation":
            validation_result,

        "risk":
            risk_result,

        "confidence":
            confidence,

        "document_consistency":
            document_consistency_result,

        "explanation":
            explanation
    }

    return render_template(

        "result.html",

        claim=
            current_claim,

        claim_chat_history=
            claim_chat_history
    )



# =========================================================
# DOWNLOADABLE CLAIM REPORT
# =========================================================

def _pdf_currency_safe(value):
    """
    Reportlab's default core font (Helvetica) does not include
    a glyph for the Indian Rupee symbol (₹), so any ₹ character
    silently fails to render in the generated PDF - it just
    disappears, leaving the number with no currency marker in
    front of it.

    This is purely a PDF rendering fix: it swaps ₹ for the
    plain-text "INR" prefix ONLY when building the PDF report.
    It does not change validate_claim / check_claim_form_vs_medical_bill
    or anything shown on the live results page (resultq.html),
    where ₹ still renders correctly in the browser.
    """

    if value is None:

        return value

    return str(value).replace("₹", "INR ")


def create_claim_report_pdf(claim):
    """
    Creates a downloadable PDF report from the final in-memory
    claim assessment. This does not change the claim calculation.
    """

    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        KeepTogether
    )

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=18,
        leading=22,
        spaceAfter=12
    )

    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        spaceBefore=10,
        spaceAfter=7
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=13,
        spaceAfter=5
    )

    small_style = ParagraphStyle(
        "ReportSmall",
        parent=styles["BodyText"],
        fontSize=8,
        leading=11
    )

    story = []

    story.append(
        Paragraph(
            "AI Hospitalization Insurance Claim Assessment Report",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Preliminary AI-assisted assessment — not a final insurance approval or rejection.",
            body_style
        )
    )

    validation = claim.get("validation", {})
    risk = claim.get("risk", {})
    claim_data = claim.get("claim", {})
    claim_bill = validation.get("claim_bill_consistency", {})

    story.append(Paragraph("1. Overall Assessment", heading_style))

    overall_data = [
        ["Claim Status", str(validation.get("overall_status", "N/A"))],
        ["Claim Risk Score", f'{risk.get("score", "N/A")}/100'],
        ["Risk Level", str(risk.get("level", "N/A"))],
        ["Evidence Quality", f'{risk.get("evidence_quality", "N/A")}/100'],
        ["Confidence", f'{claim.get("confidence", "N/A")}%']
    ]

    table = Table(overall_data, colWidths=[55 * mm, 115 * mm])
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(table)

    # -----------------------------------------------------
    # 2. CLAIM INFORMATION
    # -----------------------------------------------------

    story.append(Paragraph("2. Claim Information", heading_style))

    claim_amount_value = claim_data.get("claim_amount")

    # FIX #1: build the amount string with ₹ as before, then
    # sanitize it for the PDF so it always shows "INR ..."
    # instead of a missing/blank currency symbol.
    claim_amount_display = (
        _pdf_currency_safe(f'₹{claim_amount_value:,.2f}')
        if claim_amount_value is not None
        else "Not identified"
    )

    claim_rows_raw = [
        ["Patient Name", claim_data.get("patient_name", "")],
        ["Address", claim_data.get("address", "")],
        ["Claim Type", claim_data.get("claim_type", "Hospitalization")],
        ["Hospitalization Reason / Diagnosis", claim_data.get("hospitalization_reason", "")],
        ["Hospitalization Date", claim_data.get("claim_date", "")],
        ["Hospital / Medical Facility", claim_data.get("medical_facility", "")],
        ["Claim Amount", claim_amount_display],
        ["Description", claim_data.get("description", "")]
    ]

    # Wrap every cell in a Paragraph so long values (long
    # descriptions, diagnoses, etc.) wrap onto multiple lines
    # instead of overflowing the column width.
    claim_rows = [
        [
            Paragraph(escape(str(label)), small_style),
            Paragraph(escape(str(value)), small_style)
        ]
        for label, value in claim_rows_raw
    ]

    table = Table(claim_rows, colWidths=[65 * mm, 105 * mm])
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(table)

    # -----------------------------------------------------
    # 3. CLAIM VALIDATION
    # -----------------------------------------------------

    story.append(Paragraph("3. Claim Validation", heading_style))

    validation_rows_raw = [
        [
            "Document Check",
            validation.get("document_check", {}).get("status", "N/A"),
            validation.get("document_check", {}).get("message", "")
        ],
        [
            "Policy Exclusion Check",
            validation.get("exclusion_check", {}).get("status", "N/A"),
            validation.get("exclusion_check", {}).get("message", "")
        ]
    ]

    # FIX #2: wrap every cell (especially the long "Details"
    # column) in a Paragraph so the text wraps to fit the
    # column width instead of overflowing/being clipped, the
    # same approach already used for Table 4 below.
    validation_table_rows = [["Check", "Status", "Details"]]

    for row in validation_rows_raw:
        validation_table_rows.append([
            Paragraph(escape(str(row[0])), small_style),
            Paragraph(escape(str(row[1])), small_style),
            Paragraph(escape(_pdf_currency_safe(row[2])), small_style)
        ])

    table = Table(
        validation_table_rows,
        colWidths=[50 * mm, 28 * mm, 92 * mm],
        repeatRows=1
    )
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(table)

    story.append(Paragraph(
        "4. Claim Form vs Medical Bill Consistency",
        heading_style
    ))

    story.append(
        Paragraph(
            "The following five claim-form fields are checked against the Medical Bill: "
            "Patient Name, Hospitalization Reason / Diagnosis, "
            "Hospitalization Date, Hospital / Medical Facility, and Claim Amount. "
            ,
            body_style
        )
    )

    consistency_rows = [
        ["Field", "Status", "Submitted Value", "Medical Bill Value", "Reason"]
    ]

    for check in claim_bill.get("checks", []):
        consistency_rows.append([
            str(check.get("field", "")),
            str(check.get("status", "REVIEW")),
            str(check.get("submitted_value", "")),
            str(check.get("bill_value", "")),
            str(check.get("reason", ""))
        ])

    if len(consistency_rows) == 1:
        consistency_rows.append([
            "No field-level result",
            claim_bill.get("status", "REVIEW"),
            "",
            "",
            claim_bill.get("message", "")
        ])

    # FIX #1: sanitize ₹ -> "INR " here too, since the Claim
    # Amount row's Submitted/Bill Value and Reason columns
    # contain ₹-formatted strings from check_claim_form_vs_medical_bill.
    wrapped_rows = []
    for row in consistency_rows:
        wrapped_rows.append([
            Paragraph(escape(_pdf_currency_safe(cell)), small_style)
            for cell in row
        ])

    table = Table(
        wrapped_rows,
        colWidths=[34 * mm, 20 * mm, 36 * mm, 36 * mm, 44 * mm],
        repeatRows=1
    )
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(table)

    story.append(Paragraph("5. Risk Evidence", heading_style))

    signals = risk.get("signals", {})
    signal_rows = [
        ["Evidence Signal", "Score"],
        ["Amount Consistency", f'{signals.get("amount_consistency", "N/A")}/100'],
        ["Document Completeness", f'{signals.get("document_completeness", "N/A")}/100'],
        ["Claim Form vs Medical Bill Consistency",
         f'{signals.get("claim_bill_consistency", "N/A")}/100'],
        ["Policy Exclusion Consistency", f'{signals.get("policy_consistency", "N/A")}/100']
    ]

    table = Table(signal_rows, colWidths=[100 * mm, 40 * mm])
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(table)

    story.append(Paragraph("6. Risk Reasons", heading_style))

    reasons = risk.get("why", [])
    if reasons:
        for reason in reasons:
            story.append(Paragraph(
                "• " + escape(_pdf_currency_safe(reason)),
                body_style
            ))
    else:
        story.append(Paragraph(
            "No additional risk reasons were identified.",
            body_style
        ))

    # -----------------------------------------------------
    # NOTE: the previous "7. AI Document Consistency" and
    # "8. AI Explanation" sections have both been removed
    # per request. Section 6 (Risk Reasons) is now the last
    # numbered section before the disclaimer.
    # -----------------------------------------------------

    story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            "Disclaimer: This report is generated from the submitted claim "
            "information, uploaded documents, policy checks, and AI-assisted "
            "analysis. It is a preliminary assessment and is not a final "
            "insurance decision. Human insurer review may still be required.",
            small_style
        )
    )

    doc.build(story)
    buffer.seek(0)

    return buffer


@app.route(
    "/download-report",
    methods=["GET"]
)
def download_report():

    if not current_claim:
        return "No claim report is available. Please submit a claim first.", 404

    pdf_buffer = create_claim_report_pdf(current_claim)

    return send_file(
        pdf_buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="hospitalization_claim_assessment_report.pdf"
    )


# =========================================================
# CLAIM CHAT
# =========================================================

@app.route(
    "/ask-claim",
    methods=["POST"]
)
def ask_claim():

    question = request.form.get(
        "question",
        ""
    ).strip()

    if question:

        answer = answer_claim_question(
            question
        )

        claim_chat_history.append({

            "question":
                question,

            "answer":
                answer
        })

    return render_template(

        "result.html",

        claim=
            current_claim,

        claim_chat_history=
            claim_chat_history
    )

load_existing_policy_database()
if __name__ == "__main__":
    app.run(debug=True)