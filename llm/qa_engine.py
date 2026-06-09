import os
from openai import OpenAI
from dotenv import load_dotenv
from retrieval.hybrid_search import HybridSearch

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ClinicalQAEngine:
    def __init__(self):
        print("Initializing Clinical Q&A Engine...")
        self.searcher = HybridSearch()
        print("✅ Q&A Engine ready!")
    
    def answer_question(self, question, n_chunks=5):
        """Answer a clinical question using RAG"""
        
        # Step 1: Retrieve relevant chunks
        results = self.searcher.hybrid_search(question, n_results=n_chunks)
        
        # Step 2: Build context from retrieved chunks
        context = ""
        sources = []
        for i, result in enumerate(results):
            context += f"\n--- Source {i+1}: {result['metadata']['sample_name']} ({result['metadata']['specialty']}) ---\n"
            context += result['document']
            context += "\n"
            sources.append({
                'sample_name': result['metadata']['sample_name'],
                'specialty': result['metadata']['specialty'],
                'rrf_score': result['rrf_score']
            })
        
        # Step 3: Generate answer with GPT-4o-mini
        system_prompt = """You are a clinical information assistant helping 
healthcare professionals find information in medical records.

Rules:
- Only answer based on the provided clinical notes context
- Always cite which source(s) you used
- Be precise and use medical terminology appropriately  
- If the context doesn't contain enough information, say so clearly
- Never make up medical information not present in the context
- Format your answer clearly with the answer first, then sources"""

        user_prompt = f"""Question: {question}

Clinical Notes Context:
{context}

Please answer the question based only on the provided clinical notes."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=600,
            temperature=0.1
        )
        
        answer = response.choices[0].message.content
        
        return {
            'question': question,
            'answer': answer,
            'sources': sources,
            'chunks_used': len(results)
        }
    
    def print_answer(self, result):
        print(f"\n{'='*70}")
        print(f"❓ Question: {result['question']}")
        print(f"{'='*70}")
        print(f"\n💬 Answer:\n{result['answer']}")
        print(f"\n📚 Sources used ({result['chunks_used']} chunks):")
        for i, source in enumerate(result['sources']):
            print(f"  {i+1}. {source['sample_name']} ({source['specialty']}) - Score: {source['rrf_score']:.4f}")

if __name__ == "__main__":
    qa = ClinicalQAEngine()
    
    questions = [
        "What medications are commonly prescribed after knee replacement surgery?",
        "What are the typical findings in a cardiac catheterization procedure?",
        "What follow-up care is recommended for cardiovascular patients?"
    ]
    
    for question in questions:
        result = qa.answer_question(question)
        qa.print_answer(result)
        print("\n")