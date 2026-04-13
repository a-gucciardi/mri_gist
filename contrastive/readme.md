# Core concept

├─ Train ONLY on normal, symmetric brains (dHCP)  
├─ Learn: "What does normal bilateral symmetry look like?"  
├─ Test: Anything deviating from learned normality → Asymmetric  
└─ Domain randomization: Ensures model learns anatomy, not appearance  

## One-class approach training

Train: Normal images with domain randomization
Learn: "What normal symmetry looks like across ANY appearance"
Success: Flags ANY deviation from normal, including unseen pathologies (2nd dataset)