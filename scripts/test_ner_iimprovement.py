import time
# Assuming your old function is in 'resume_ner_bert_old' and new is in 'resume_ner_bert_v2'
# If they are in the same file, rename the old one to `parse_resume_ner_bert_OLD`
from resume_ner_bert import parse_resume_ner_bert as old_ner
from resume_ner_bert_v2 import parse_resume_ner_bert as new_ner
from resume_parser_util import extract_text_from_file

SAMPLE = """
John Doe
Summary
Software engineer with 5 years of experience in Python and cloud systems.
Experience
Senior Software Engineer
Jan 2020 to Current
Tech Company Inc. — New York, NY
Built APIs and data pipelines. Led a team of 4.
Software Developer
Mar 2018 to Dec 2019
Startup Co — San Francisco, CA
Skills
Python, Java, AWS, SQL, REST APIs, Docker, Kubernetes, machine learning
Education
Bachelor of Science in Computer Science 2016
State University — Boston, MA
"""

OwnCV =r"C:\Vasanth\Important stuff\Resumes\Vasanth Subramanian Resume.pdf"
print("Running BERT resume NER (first run may download the model)...")
SAMPLE_text = extract_text_from_file(OwnCV)

test_cases= [SAMPLE_text, SAMPLE]

def main():
    for idx, text in enumerate(test_cases):
        print(f"\n\n=== TEST CASE {idx+1} ===")
        print("Evaluating Old NER Function...")
        t0 = time.time()
        old_res = old_ner(text)
        t1 = time.time()
        
        print("\nEvaluating New NER Function...")
        new_res = new_ner(text)
        t2 = time.time()
        
        print("\n" + "="*50)
        print(f"OLD NER RESULTS ({t1-t0:.2f}s):")
        for k, v in old_res.items(): print(f"  {k}: {v}")
            
        print("\n" + "="*50)
        print(f"NEW NER RESULTS ({t2-t1:.2f}s):")
        for k, v in new_res.items(): print(f"  {k}: {v}")
        print("="*50)

if __name__ == "__main__":
    main()