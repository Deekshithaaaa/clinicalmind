# 25 clinical test questions for A/B testing
TEST_QUESTIONS = [
    {
        "question": "What are the typical findings in a cardiac catheterization procedure?",
        "specialty": "Cardiovascular"
    },
    {
        "question": "What medications are prescribed after knee replacement surgery?",
        "specialty": "Orthopedic"
    },
    {
        "question": "What are the symptoms and treatment for atrial fibrillation?",
        "specialty": "Cardiovascular"
    },
    {
        "question": "What does a normal echocardiogram show?",
        "specialty": "Cardiovascular"
    },
    {
        "question": "What are the preoperative and postoperative diagnoses for knee arthroplasty?",
        "specialty": "Orthopedic"
    },
    {
        "question": "What anesthesia is used during orthopedic surgery?",
        "specialty": "Orthopedic"
    },
    {
        "question": "What are common findings in a radiology CT scan of the abdomen?",
        "specialty": "Radiology"
    },
    {
        "question": "What is the procedure for a laparoscopic surgery?",
        "specialty": "Surgery"
    },
    {
        "question": "What medications are used for hypertension management?",
        "specialty": "General Medicine"
    },
    {
        "question": "What are the indications for a pacemaker implantation?",
        "specialty": "Cardiovascular"
    },
    {
        "question": "What are the findings in a neurology consultation?",
        "specialty": "Neurology"
    },
    {
        "question": "What is the treatment plan for a patient with chest pain?",
        "specialty": "Cardiovascular"
    },
    {
        "question": "What are the key findings in an orthopedic examination?",
        "specialty": "Orthopedic"
    },
    {
        "question": "What medications are prescribed for pain management after surgery?",
        "specialty": "Surgery"
    },
    {
        "question": "What are the typical symptoms of gastroenterology conditions?",
        "specialty": "Gastroenterology"
    },
    {
        "question": "What does a discharge summary typically include?",
        "specialty": "General Medicine"
    },
    {
        "question": "What are the findings in a pulmonary function test?",
        "specialty": "Cardiovascular"
    },
    {
        "question": "What is the procedure for coronary angioplasty?",
        "specialty": "Cardiovascular"
    },
    {
        "question": "What are the postoperative care instructions for surgery patients?",
        "specialty": "Surgery"
    },
    {
        "question": "What are the common diagnoses in general medicine consultations?",
        "specialty": "General Medicine"
    },
    {
        "question": "What imaging studies are ordered for back pain?",
        "specialty": "Radiology"
    },
    {
        "question": "What are the findings in a neurosurgery consultation?",
        "specialty": "Neurosurgery"
    },
    {
        "question": "What medications are used in emergency room treatment?",
        "specialty": "Surgery"
    },
    {
        "question": "What are the typical findings in an obstetrics delivery note?",
        "specialty": "Obstetrics"
    },
    {
        "question": "What is the recovery process after spinal surgery?",
        "specialty": "Neurosurgery"
    }
]

if __name__ == "__main__":
    print(f"Total test questions: {len(TEST_QUESTIONS)}")
    for i, q in enumerate(TEST_QUESTIONS):
        print(f"{i+1}. [{q['specialty']}] {q['question']}")
