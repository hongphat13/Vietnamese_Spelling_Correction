
from fastapi import FastAPI
from pydantic import BaseModel
from src.hmm_decoder import HMMSpellChecker

app = FastAPI(title="Viet Spell API")
checker = HMMSpellChecker.from_artifacts("artifacts")

class Req(BaseModel):
    text: str

@app.post("/correct")
def correct(req: Req):
    out = checker.correct(req.text)
    return {"input": req.text, "corrected": out}
