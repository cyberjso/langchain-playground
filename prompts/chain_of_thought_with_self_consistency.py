"""
This prompt style encourages the model to think step-by-step and provide intermediate reasoning or explanations before arriving at a final answer.
It also incorporates self-consistency by generating multiple reasoning paths and selecting the most consistent final answer.

Useful in the following scenarios:
- When the task requires multi-step reasoning or problem-solving.
- When looking for more accurate and reliable answers by leveraging multiple reasoning paths.
- Complex mathematical calculations or logical reasoning.
- Tradeoff analysis or decision-making tasks where multiple perspectives are beneficial.

Negative scenarios to avoid using this prompt style:
- The task is simple and can be answered directly without additional reasoning.
- Minimize token usage costs
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

prompt = """ 
You are an expert Credit Analyst for a major auto lending institution. Your task is to perform a comprehensive hypothetical credit analysis for a prospective car buyer. 
You must evaluate the applicant's financial health, determine the risk level, and recommend the best loan structure (or reasons for denial).

**APPLICANT PROFILE AND FINANCIAL DATA:**
* **Credit Score (FICO):** 715
* **Employment Status:** Full-Time, Salaried (5 years at current company)
* **Annual Gross Income:** $90,000
* **Monthly Gross Income:** $7,500 ($90,000 / 12)
* **Total Existing Monthly Debt Payments:** $1,100 (Includes credit cards, student loans, and a personal loan)
* **Monthly Rent/Mortgage Payment:** $1,800
* **Savings/Emergency Fund (Liquid Assets):** $15,000
* **Target Vehicle Price (MSRP):** $35,000
* **Down Payment Amount:** $5,000
* **Loan Term Requested:** 60 Months
* **Desired APR Range (For Evaluation):** 4.5% - 6.5%

**ANALYSIS INSTRUCTIONS:**

Generate 3 different reasoning paths step by step to analyze the applicant's creditworthiness based on the provided data.
At the end, summarize the findings from each path and provide a final recommendation on loan approval, structure, or denial based on the most consistent insights across the paths.
"""

model = ChatOpenAI(model="gpt-4o", temperature=0)
message = ChatPromptTemplate([prompt]).format_messages()
result_chat = model.invoke(message)

print(f"Response:\n{result_chat.content}\n")
print(f"tokens used: {result_chat.response_metadata['token_usage']}\n")