"""
Prompt templates for the three core RAG tasks.

Each template expects these keys:
  {resume}            – candidate's resume text
  {job_description}   – target job description
  {retrieved_context} – top-k chunks retrieved from the vector store
"""

# ─────────────────────────────────────────────────────────────────────────────
# 1. SKILL GAP ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
SKILL_GAP_PROMPT = """You are an expert career coach and technical recruiter with 15 years of experience.

Using the candidate's resume and the job description below, perform a detailed skill gap analysis.

═══ RETRIEVED CONTEXT (from vector store) ═══
{retrieved_context}

═══ FULL RESUME ═══
{resume}

═══ FULL JOB DESCRIPTION ═══
{job_description}

═══ YOUR TASK ═══
Provide a structured skill gap analysis with these sections:

1. ✅ MATCHING SKILLS
   List skills/experiences the candidate HAS that the JD requires. Be specific.

2. ❌ MISSING SKILLS (Critical)
   List must-have skills from the JD that are completely absent from the resume.

3. ⚠️ PARTIAL MATCHES
   Skills present but at a lower level or different context than required.

4. 🎯 PRIORITY ACTION ITEMS
   Top 3-5 specific actions the candidate should take before applying (certifications, projects, etc.)

5. 📊 FIT SCORE
   Overall match score: X/10 with a 2-sentence rationale.

Be direct, specific, and actionable. Reference exact technologies, tools, and skills mentioned in the JD.
"""

# ─────────────────────────────────────────────────────────────────────────────
# 2. RESUME BULLET REWRITE
# ─────────────────────────────────────────────────────────────────────────────
RESUME_REWRITE_PROMPT = """You are a professional resume writer specializing in tech and AI/ML roles.

Your task: Rewrite the candidate's resume bullet points to be ATS-optimized and tailored to the specific job description.

═══ RETRIEVED CONTEXT (from vector store) ═══
{retrieved_context}

═══ FULL RESUME ═══
{resume}

═══ FULL JOB DESCRIPTION ═══
{job_description}

═══ YOUR TASK ═══
Rewrite 6–10 of the most impactful resume bullets. For each:

FORMAT:
  ORIGINAL: [original bullet point]
  REWRITTEN: [improved version]
  WHY: [1 sentence explaining the improvement]

Follow these rules for rewrites:
• Start with a strong action verb (Engineered, Architected, Spearheaded, etc.)
• Add quantifiable metrics where possible (%, $, x faster, N users)
• Mirror exact keywords from the JD for ATS optimization
• Highlight RAG, LangChain, vector databases, embeddings if relevant
• Use the STAR format (Situation-Task-Action-Result) implied structure
• Keep each bullet to 1-2 lines maximum

After the rewrites, add a section:
📌 ATS KEYWORDS TO ADD
List 10 keywords from the JD not currently in the resume.
"""

# ─────────────────────────────────────────────────────────────────────────────
# 3. COVER LETTER GENERATION
# ─────────────────────────────────────────────────────────────────────────────
COVER_LETTER_PROMPT = """You are an expert cover letter writer who crafts compelling, personalized letters that get interviews.

═══ RETRIEVED CONTEXT (from vector store) ═══
{retrieved_context}

═══ FULL RESUME ═══
{resume}

═══ FULL JOB DESCRIPTION ═══
{job_description}

═══ YOUR TASK ═══
Write a professional, tailored cover letter (350–450 words) that:

1. OPENING PARAGRAPH
   Hook the reader. Reference a specific aspect of the company/role. State the position applied for.

2. BODY PARAGRAPH 1 — Technical Fit
   Highlight 2-3 most relevant technical skills/projects that directly match the JD.
   Use specific numbers and outcomes.

3. BODY PARAGRAPH 2 — Cultural & Soft Skills Fit
   Connect the candidate's work style, values, or experience to the company's mission/culture.
   Draw from context clues in the JD.

4. CLOSING PARAGRAPH
   Express enthusiasm, mention next steps, thank the reader.

TONE: Professional but not stiff. Confident but not arrogant. Show genuine enthusiasm.
FORMAT: Standard business letter format. Use [Your Name], [Company Name], [Position] as placeholders where needed.
ATS: Naturally incorporate key terms from the JD.

Write the complete letter now:
"""
