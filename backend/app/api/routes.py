from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import Optional
from app.agent.orchestrator import MedicalAgentProcess

router = APIRouter()

@router.post("/analyze")
async def analyze_patient_data(
    biomarkers_json: Optional[str] = Form(None),
    prescription_image: Optional[UploadFile] = File(None),
    posture_video: Optional[UploadFile] = File(None)
):
    """
    Main endpoint that accepts multimodel inputs and routes them through the Medical AI Co-Pilot.
    """
    try:
        # Example of handling uploaded files (in reality, save to temp storage/S3)
        prescription_path = None
        posture_video_path = None
        
        if prescription_image:
            prescription_path = f"/tmp/{prescription_image.filename}"
            with open(prescription_path, "wb") as f:
                content = await prescription_image.read()
                f.write(content)
                
        if posture_video:
            posture_video_path = f"/tmp/{posture_video.filename}"
            with open(posture_video_path, "wb") as f:
                content = await posture_video.read()
                f.write(content)

        # Initialize the Medical Agent
        agent = MedicalAgentProcess()
        
        # Execute the orchestration
        analysis_result = await agent.run_analysis(
            biomarkers=biomarkers_json,
            prescription_path=prescription_path,
            posture_video_path=posture_video_path
        )
        
        return {
            "status": "success",
            "data": analysis_result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
