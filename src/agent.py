"""
Agentic AI Retention Strategy Assistant
========================================
Uses LangGraph for workflow orchestration and FAISS for RAG-based strategy retrieval.
Generates structured retention reports using OpenRouter.
"""

import os
from typing import TypedDict, List, Optional
import numpy as np
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer

from dotenv import load_dotenv
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class TfidfEmbeddings(Embeddings):
    """Local TF-IDF embeddings using sklearn — runs entirely offline, no API calls."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=256,
            stop_words="english",
            ngram_range=(1, 2),
        )
        self._fitted = False

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = self.vectorizer.fit_transform(texts).toarray()
        self._fitted = True
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        if not self._fitted:
            return [0.0] * 256
        vector = self.vectorizer.transform([text]).toarray()[0]
        return vector.tolist()


class AgentState(TypedDict):
    """Explicit state management across all workflow steps."""
    customer_profile: dict
    churn_probability: float
    churn_risk_level: str
    drivers: List[str]
    retrieved_strategies: str
    report: str
    error: Optional[str]
    api_key: Optional[str]


def _load_knowledge_base() -> List[Document]:
    """Load and parse the retention strategies knowledge base into documents."""
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "retention_strategies.txt")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    chunks = content.split("## Strategy:")
    docs = []
    for chunk in chunks:
        chunk = chunk.strip()
        if chunk and len(chunk) > 50:
            docs.append(Document(
                page_content=f"Retention Strategy: {chunk}",
                metadata={"source": "retention_strategies.txt"}
            ))
    return docs


def analyze_risk(state: AgentState) -> AgentState:
    """Analyze the customer profile and determine risk level with reasoning."""
    profile = state.get("customer_profile", {})
    prob = state.get("churn_probability", 0.0)

    if prob >= 0.6:
        state["churn_risk_level"] = "High Risk"
    elif prob >= 0.35:
        state["churn_risk_level"] = "Medium Risk"
    else:
        state["churn_risk_level"] = "Low Risk"

    drivers = []
    tenure = profile.get("tenure", 0)
    monthly = profile.get("monthly_charges", 0)
    support = profile.get("support_calls", 0)
    sentiment_label = profile.get("sentiment_label", "neutral")
    issue_label = profile.get("issue_label", "general")

    if tenure < 12:
        drivers.append(f"Early tenure ({tenure} months) — high first-year attrition risk")
    elif tenure < 24:
        drivers.append(f"Moderate tenure ({tenure} months) — re-engagement window")
    else:
        drivers.append(f"Long-term customer ({tenure} months) — loyalty preservation needed")

    if monthly > 80:
        drivers.append(f"High monthly charges (${monthly:.0f}/mo) — cost sensitivity pressure")
    elif monthly > 50:
        drivers.append(f"Moderate monthly charges (${monthly:.0f}/mo) — value perception risk")
    else:
        drivers.append(f"Low monthly charges (${monthly:.0f}/mo) — affordable tier")

    if support >= 3:
        drivers.append(f"Frequent support contacts ({support} calls) — deep dissatisfaction signal")
    elif support >= 1:
        drivers.append(f"Some support contacts ({support} call(s)) — friction indicator")
    else:
        drivers.append("No support contacts — no friction signals")

    if sentiment_label == "negative":
        drivers.append(f"NLP sentiment is negative with issue category '{issue_label}'")
    elif sentiment_label == "neutral":
        drivers.append(f"NLP sentiment is neutral with issue category '{issue_label}'")
    else:
        drivers.append(f"NLP sentiment is positive with issue category '{issue_label}'")

    state["drivers"] = drivers
    return state


def retrieve_strategies(state: AgentState) -> AgentState:
    """Use FAISS vector store + local TF-IDF embeddings to retrieve relevant retention strategies."""
    if state.get("error"):
        return state

    try:
        embeddings = TfidfEmbeddings()

        docs = _load_knowledge_base()
        if not docs:
            state["retrieved_strategies"] = "No retention knowledge base found."
            return state

        vectorstore = FAISS.from_documents(docs, embeddings)

        profile = state.get("customer_profile", {})
        query_parts = state.get("drivers", [])
        query = " ".join(query_parts)
        query += f" Customer tenure: {profile.get('tenure', 0)} months."
        query += f" Monthly charges: ${profile.get('monthly_charges', 0)}."
        query += f" Churn probability: {state.get('churn_probability', 0):.0%}."
        query += f" Sentiment: {profile.get('sentiment_label', 'neutral')}."
        query += f" Issue category: {profile.get('issue_label', 'general')}."
        if profile.get("support_text"):
            query += f" Ticket text: {profile.get('support_text')[:220]}."

        results = vectorstore.similarity_search(query, k=3)
        strategies_text = "\n\n---\n\n".join([doc.page_content for doc in results])
        state["retrieved_strategies"] = strategies_text

        return state

    except Exception as e:
        state["error"] = f"RAG Retrieval Error: {str(e)}"
        return state


def generate_report(state: AgentState) -> AgentState:
    """Generate structured retention report using OpenRouter LLM with anti-hallucination prompting."""
    if state.get("error"):
        return state

    api_key = state.get("api_key") or OPENROUTER_API_KEY
    if not api_key or len(api_key) < 10:
        try:
            import streamlit as _st
            api_key = _st.secrets.get("OPENROUTER_API_KEY")
        except Exception:
            pass
    if not api_key or len(api_key) < 10:
        state["error"] = "API Key Error: Please provide a valid OpenRouter API Key in the sidebar configuration."
        return state

    try:
        llm = ChatOpenAI(
            model="anthropic/claude-3-haiku",
            openai_api_key=api_key,
            openai_api_base=OPENROUTER_BASE_URL,
            temperature=0.15,
            default_headers={"HTTP-Referer": "http://localhost:8501", "X-Title": "ChurnSense"}
        )

        profile = state.get("customer_profile", {})
        risk = state.get("churn_risk_level", "Unknown")
        prob = state.get("churn_probability", 0.0)
        drivers = state.get("drivers", [])
        strategies = state.get("retrieved_strategies", "")

        prompt = f"""You are a Customer Retention AI Analyst. Your task is to generate a structured retention report.

IMPORTANT INSTRUCTIONS:
- Base ALL recommendations ONLY on the retrieved strategies and customer data provided below.
- Do NOT invent statistics, studies, or data points not present in the provided context.
- If information is insufficient, explicitly state what additional data would be needed.
- Use professional, actionable language suitable for a business audience.

═══ CUSTOMER DATA ═══
• Risk Level: {risk}
• Churn Probability: {prob*100:.1f}%
• Tenure: {profile.get('tenure', 'N/A')} months
• Monthly Charges: ${profile.get('monthly_charges', 'N/A')}
• Total Charges: ${profile.get('total_charges', 'N/A')}
• Support Calls: {profile.get('support_calls', 'N/A')}
• Avg Monthly Spend: ${profile.get('avg_monthly_spend', 'N/A')}
• NLP Sentiment: {profile.get('sentiment_label', 'N/A')}
• NLP Issue Category: {profile.get('issue_label', 'N/A')}
• Ticket/Feedback Snippet: {str(profile.get('support_text', 'N/A'))[:220]}

═══ IDENTIFIED RISK DRIVERS ═══
{chr(10).join(f'• {d}' for d in drivers)}

═══ RETRIEVED RETENTION STRATEGIES (from knowledge base) ═══
{strategies}

═══ REQUIRED OUTPUT FORMAT ═══
Generate the report in Markdown with these EXACT sections:

### 🔍 Risk Summary
Provide a 2-3 sentence profile of this customer's churn risk, citing the specific data points above.

### 💡 Recommendations
Provide 3 specific, actionable retention interventions drawn from the retrieved strategies. For each:
- **Action [N]**: [Title]
  - What to do and why, tied to this customer's specific profile.

### 📚 Sources
List the specific knowledge base strategies and references that informed your recommendations.

### ⚖️ Disclaimer
Standard disclaimer that these are AI-generated recommendations that should be reviewed by a human retention specialist before implementation. Mention that individual customer circumstances may vary and ethical considerations should guide all retention efforts.
"""

        response = llm.invoke(prompt)
        state["report"] = response.content
        return state

    except Exception as e:
        state["error"] = f"Report Generation Error: {str(e)}"
        return state


def build_retention_agent():
    """Construct and compile the LangGraph state machine for retention analysis."""
    workflow = StateGraph(AgentState)

    workflow.add_node("analyze_risk", analyze_risk)
    workflow.add_node("retrieve_strategies", retrieve_strategies)
    workflow.add_node("generate_report", generate_report)

    workflow.set_entry_point("analyze_risk")
    workflow.add_edge("analyze_risk", "retrieve_strategies")
    workflow.add_edge("retrieve_strategies", "generate_report")
    workflow.add_edge("generate_report", END)

    return workflow.compile()
