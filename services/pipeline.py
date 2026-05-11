from services.ml_service import get_stage
from services.rag_service import get_medical_context
from services.llm_service import get_ai_response

def run_pipeline(egfr):
    stage = get_stage(egfr)

    context = get_medical_context(stage)

    prompt = f"""
    Patient CKD Stage: {stage}

    Medical Data:
    {context}

    Explain in simple doctor language:
    - disease
    - diet
    - treatment
    - precautions
    """

    response = get_ai_response(prompt)

    return stage, response