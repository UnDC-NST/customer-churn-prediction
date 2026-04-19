from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

load_dotenv()

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_groq import ChatGroq
except Exception:
    ChatGroq = None
    HumanMessage = None
    SystemMessage = None

from preprocess import columns as training_columns
from retention_knowledge_base import RETENTION_DOCUMENTS, RETENTION_SOURCES


PROTECTED_FACTORS = {"gender", "seniorcitizen", "age", "race", "religion"}


class RetentionState(TypedDict, total=False):
    customer_profile: Dict[str, Any]
    processed_features: Dict[str, float]
    churn_probability: float
    churn_prediction: int
    risk_level: str
    data_quality_notes: List[str]
    retrieval_query: str
    key_factors: List[Dict[str, Any]]
    retrieved_practices: List[Dict[str, Any]]
    supporting_sources: List[Dict[str, Any]]
    draft_report: Dict[str, Any]
    reviewed_report: Dict[str, Any]
    final_report: Dict[str, Any]
    warnings: List[str]


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == "":
            return default
        return float(value)
    except Exception:
        return default


def _risk_label(probability: float) -> str:
    if probability < 0.3:
        return "Low"
    if probability < 0.6:
        return "Medium"
    return "High"


class AgenticRetentionAssistant:
    def __init__(
        self,
        model: Any,
        llm_model: str = "llama-3.1-8b-instant",
        groq_api_key: Optional[str] = None,
    ) -> None:
        self.model = model
        self.llm_model = llm_model
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")

        self.coefficients = self._extract_coefficients(model)
        self.feature_names = self._extract_feature_names(model, len(self.coefficients))
        self.coef_map = {feature: float(coef) for feature, coef in zip(self.feature_names, self.coefficients)}

        self.sources_by_id = {item["id"]: item for item in RETENTION_SOURCES}
        self.retrieval_docs = list(RETENTION_DOCUMENTS)

        self._retrieval_vectorizer: Optional[TfidfVectorizer] = None
        self._retrieval_matrix = None
        self._build_retrieval_index()

        self.graph = self._build_graph()

    def _build_retrieval_index(self) -> None:
        if not self.retrieval_docs:
            return

        texts = [self._compose_doc_text(doc) for doc in self.retrieval_docs]
        self._retrieval_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._retrieval_matrix = self._retrieval_vectorizer.fit_transform(texts)

    def _compose_doc_text(self, doc: Dict[str, Any]) -> str:
        return f"{doc.get('title', '')}. {doc.get('content', '')}".strip()

    def _extract_coefficients(self, model: Any) -> np.ndarray:
        estimator = model
        if hasattr(model, "named_steps"):
            for key in ["logisticregression", "classifier", "model", "clf", "lr"]:
                if key in model.named_steps:
                    estimator = model.named_steps[key]
                    break
            else:
                estimator = list(model.named_steps.values())[-1]

        coef = np.ravel(getattr(estimator, "coef_", np.array([])))
        if coef.size == 0:
            raise ValueError("Unable to extract model coefficients for factor analysis.")
        return coef

    def _extract_feature_names(self, model: Any, coef_len: int) -> List[str]:
        model_names = getattr(model, "feature_names_in_", None)
        if model_names is not None and len(model_names) == coef_len:
            return list(model_names)

        if len(training_columns) == coef_len:
            return list(training_columns)

        return [f"feature_{idx}" for idx in range(coef_len)]

    def _build_graph(self):
        workflow = StateGraph(RetentionState)
        workflow.add_node("sanitize_step", self._node_sanitize_input)
        workflow.add_node("analyze_step", self._node_analyze_factors)
        workflow.add_node("retrieve_step", self._node_retrieve_practices)
        workflow.add_node("draft_step", self._node_draft_report)
        workflow.add_node("review_step", self._node_self_review)
        workflow.add_node("safety_step", self._node_safety_gate)
        workflow.add_node("finalize_step", self._node_finalize)

        workflow.set_entry_point("sanitize_step")
        workflow.add_edge("sanitize_step", "analyze_step")
        workflow.add_edge("analyze_step", "retrieve_step")
        workflow.add_edge("retrieve_step", "draft_step")
        workflow.add_edge("draft_step", "review_step")
        workflow.add_edge("review_step", "safety_step")
        workflow.add_edge("safety_step", "finalize_step")
        workflow.add_edge("finalize_step", END)

        return workflow.compile()

    def _llm(self):
        if not self.groq_api_key or ChatGroq is None:
            return None
        return ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name=self.llm_model,
            temperature=0.1,
        )

    def _append_warning(self, state: RetentionState, message: str) -> List[str]:
        warnings = list(state.get("warnings", []))
        if message not in warnings:
            warnings.append(message)
        return warnings

    def _default_non_generated_segment(self) -> str:
        return "LLM output unavailable"

    def _default_non_generated_strategy(self) -> str:
        return "No strategy generated because valid LLM output was unavailable."

    def _format_supporting_sources(self, state: RetentionState, max_items: int = 4) -> List[Dict[str, Any]]:
        source_pool = state.get("supporting_sources", [])
        if not source_pool:
            source_pool = self._sources_from_docs(state.get("retrieved_practices", []))

        formatted: List[Dict[str, Any]] = []
        for source in source_pool[:max_items]:
            formatted.append(
                {
                    "id": source.get("id", ""),
                    "title": source.get("title", ""),
                    "url": source.get("url", ""),
                    "justification": source.get(
                        "justification",
                        "Retrieved as relevant context for this customer profile.",
                    ),
                }
            )

        return formatted

    def _llm_only_unavailable_report(self, state: RetentionState, reason: str) -> Dict[str, Any]:
        risk = state.get("risk_level", _risk_label(float(state.get("churn_probability", 0.0))))
        probability_pct = round(float(state.get("churn_probability", 0.0)) * 100, 2)
        factors = [item.get("explanation", "") for item in state.get("key_factors", [])[:6]]

        return {
            "churn_risk_summary": (
                f"Predicted churn risk is {risk} at {probability_pct}%. "
                "LLM-only mode is enabled, so retention suggestions are shown only when a valid LLM response is available."
            ),
            "customer_segment": self._default_non_generated_segment(),
            "segment_strategy": self._default_non_generated_strategy(),
            "key_contributing_factors": factors,
            "recommended_retention_actions": [],
            "supporting_sources": self._format_supporting_sources(state),
            "business_and_ethical_disclaimers": [
                "No deterministic fallback suggestions were generated because LLM-only mode is active.",
                f"Generation note: {reason}",
                "Recommendations must comply with privacy, consent, and non-discrimination policies.",
            ],
        }

    def _semantic_retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not query or self._retrieval_vectorizer is None or self._retrieval_matrix is None:
            return []

        query_vector = self._retrieval_vectorizer.transform([query])
        scores = (self._retrieval_matrix @ query_vector.T).toarray().ravel()

        if scores.size == 0:
            return []

        top_indices = np.argsort(scores)[::-1][:top_k]
        retrieved = []
        for idx in top_indices:
            doc = dict(self.retrieval_docs[idx])
            doc["retrieval_score"] = float(scores[idx])
            retrieved.append(doc)

        return retrieved

    def _sources_from_docs(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        source_ids: List[str] = []
        for doc in docs:
            source_ids.extend(doc.get("source_ids", []))

        unique_sources: List[Dict[str, Any]] = []
        for source_id in dict.fromkeys(source_ids):
            source = self.sources_by_id.get(source_id)
            if source:
                unique_sources.append(source)

        return unique_sources

    def _build_retrieval_query(
        self,
        profile: Dict[str, Any],
        risk_level: str,
        factors: List[Dict[str, Any]],
    ) -> str:
        profile_keys = [
            "Contract", "PaymentMethod", "InternetService", "tenure",
            "MonthlyCharges", "PaperlessBilling", "OnlineSecurity", "TechSupport",
        ]
        profile_summary = []
        for key in profile_keys:
            if key in profile:
                profile_summary.append(f"{key}: {profile.get(key)}")

        factor_summary = [item.get("explanation", "") for item in factors[:5]]

        return (
            f"Risk level: {risk_level}. "
            f"Profile context: {' | '.join(profile_summary)}. "
            f"Model factors: {' | '.join(factor_summary)}. "
            "Retrieve telecom retention best practices relevant to this customer."
        )

    def _node_sanitize_input(self, state: RetentionState) -> RetentionState:
        profile = state.get("customer_profile", {})
        probability = float(state.get("churn_probability", 0.0))
        risk_level = _risk_label(probability)
        quality_notes = self._assess_data_quality(profile)

        retrieval_query = self._build_retrieval_query(profile, risk_level, [])

        return {
            "risk_level": risk_level,
            "data_quality_notes": quality_notes,
            "retrieval_query": retrieval_query,
            "warnings": [] if not quality_notes else ["Input data had missing/noisy values and was normalized."],
        }

    def _node_analyze_factors(self, state: RetentionState) -> RetentionState:
        processed = state.get("processed_features", {})
        customer_profile = state.get("customer_profile", {})

        contributions: List[Dict[str, Any]] = []
        for feature, coef in self.coef_map.items():
            value = _to_float(processed.get(feature, 0.0), 0.0)
            contribution = coef * value
            if abs(contribution) < 1e-9:
                continue

            contributions.append(
                {
                    "feature": feature,
                    "contribution": float(contribution),
                    "direction": "increases" if contribution > 0 else "reduces",
                    "explanation": self._explain_feature(feature, contribution, customer_profile),
                }
            )

        contributions = sorted(contributions, key=lambda x: abs(x["contribution"]), reverse=True)
        key_factors = contributions[:6]

        retrieval_query = self._build_retrieval_query(
            customer_profile,
            state.get("risk_level", "Medium"),
            key_factors,
        )

        return {
            "key_factors": key_factors,
            "retrieval_query": retrieval_query,
        }

    def _node_retrieve_practices(self, state: RetentionState) -> RetentionState:
        query = state.get("retrieval_query", "")
        retrieved_docs = self._semantic_retrieve(query, top_k=6)
        supporting_sources = self._sources_from_docs(retrieved_docs)

        return {
            "retrieved_practices": retrieved_docs,
            "supporting_sources": supporting_sources,
        }

    def _node_draft_report(self, state: RetentionState) -> RetentionState:
        llm = self._llm()
        if llm is None:
            warning = "LLM-only mode: no active LLM connection, so suggestions were not generated."
            return {
                "draft_report": self._llm_only_unavailable_report(state, warning),
                "warnings": self._append_warning(state, warning),
            }

        prompt_payload = {
            "risk_level": state.get("risk_level"),
            "churn_probability": round(float(state.get("churn_probability", 0.0)) * 100, 2),
            "customer_profile": state.get("customer_profile", {}),
            "data_quality_notes": state.get("data_quality_notes", []),
            "key_factors": [factor.get("explanation", "") for factor in state.get("key_factors", [])],
            "retrieved_context": [
                {
                    "title": doc.get("title", ""),
                    "content": doc.get("content", ""),
                    "retrieval_score": doc.get("retrieval_score", 0.0),
                    "source_ids": doc.get("source_ids", []),
                }
                for doc in state.get("retrieved_practices", [])
            ],
            "supporting_sources": state.get("supporting_sources", []),
        }

        system_prompt = (
            "You are an agentic telecom retention strategist.\n"
            "Use only provided retrieved_context and supporting_sources as evidence.\n"
            "Do not invent source IDs, URLs, or factual claims.\n"
            "Avoid discriminatory or biased recommendations and avoid certainty language.\n"
            "Return strict JSON with keys:\n"
            "churn_risk_summary (string),\n"
            "customer_segment (string),\n"
            "segment_strategy (string),\n"
            "key_contributing_factors (array of strings),\n"
            "recommended_retention_actions (array of objects with action, rationale, priority, execution_notes),\n"
            "supporting_sources (array of objects with id,title,url,justification),\n"
            "business_and_ethical_disclaimers (array of strings)."
        )

        human_prompt = f"Create a grounded retention report from this context: {json.dumps(prompt_payload)}"

        try:
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ])
            report = self._parse_json_object(response.content)
            if not report:
                warning = "LLM returned invalid JSON in draft step, so suggestions were withheld."
                return {
                    "draft_report": self._llm_only_unavailable_report(state, warning),
                    "warnings": self._append_warning(state, warning),
                }
        except Exception:
            warning = "LLM invocation failed in draft step, so suggestions were withheld."
            return {
                "draft_report": self._llm_only_unavailable_report(state, warning),
                "warnings": self._append_warning(state, warning),
            }

        return {"draft_report": self._ensure_report_schema(report)}

    def _node_self_review(self, state: RetentionState) -> RetentionState:
        draft_report = state.get("draft_report", {})
        llm = self._llm()
        if llm is None:
            return {"reviewed_report": self._ensure_report_schema(draft_report)}

        system_prompt = (
            "You are a strict reviewer for groundedness and ethics.\n"
            "Revise the report so all recommendations are supported by provided sources and factors.\n"
            "Remove unsupported or biased suggestions.\n"
            "Return strict JSON with the same schema."
        )

        review_context = {
            "report": draft_report,
            "allowed_sources": state.get("supporting_sources", []),
            "allowed_factors": [factor.get("explanation", "") for factor in state.get("key_factors", [])],
            "data_quality_notes": state.get("data_quality_notes", []),
        }

        try:
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=json.dumps(review_context)),
            ])
            reviewed = self._parse_json_object(response.content)
            if not reviewed:
                warning = "LLM self-review returned invalid JSON; using sanitized draft without fallback generation."
                return {
                    "reviewed_report": self._apply_rule_based_review(draft_report, state),
                    "warnings": self._append_warning(state, warning),
                }
        except Exception:
            warning = "LLM self-review failed; using sanitized draft without fallback generation."
            return {
                "reviewed_report": self._apply_rule_based_review(draft_report, state),
                "warnings": self._append_warning(state, warning),
            }

        return {"reviewed_report": self._apply_rule_based_review(reviewed, state)}

    def _node_safety_gate(self, state: RetentionState) -> RetentionState:
        report = self._ensure_report_schema(state.get("reviewed_report", {}))
        warnings = list(state.get("warnings", []))

        filtered_actions = []
        for action_item in report.get("recommended_retention_actions", []):
            text_blob = " ".join([
                str(action_item.get("action", "")),
                str(action_item.get("rationale", "")),
                str(action_item.get("execution_notes", "")),
            ]).lower()
            if any(token in text_blob for token in PROTECTED_FACTORS):
                continue
            filtered_actions.append(action_item)

        if report.get("recommended_retention_actions") and not filtered_actions:
            warning = "All generated retention actions were removed by safety filters."
            if warning not in warnings:
                warnings.append(warning)

        report["recommended_retention_actions"] = filtered_actions[:5]

        if not report.get("supporting_sources"):
            report["supporting_sources"] = self._format_supporting_sources(state)

        disclaimers = report.get("business_and_ethical_disclaimers", [])
        default_disclaimers = [
            "This report is decision support and should not be the sole basis for customer actions.",
            "Recommendations must comply with privacy, consent, and non-discrimination policies.",
            "Do not target or exclude customers using protected attributes.",
            "Validate expected business impact with controlled experiments before full rollout.",
        ]
        for disclaimer in default_disclaimers:
            if disclaimer not in disclaimers:
                disclaimers.append(disclaimer)

        for note in state.get("data_quality_notes", []):
            data_note = f"Data quality note: {note}"
            if data_note not in disclaimers:
                disclaimers.append(data_note)

        report["business_and_ethical_disclaimers"] = disclaimers

        if not str(report.get("customer_segment", "")).strip():
            report["customer_segment"] = self._default_non_generated_segment()
        if not str(report.get("segment_strategy", "")).strip():
            report["segment_strategy"] = self._default_non_generated_strategy()

        return {"reviewed_report": report, "warnings": warnings}

    def _node_finalize(self, state: RetentionState) -> RetentionState:
        report = self._ensure_report_schema(state.get("reviewed_report", {}))

        if not str(report.get("customer_segment", "")).strip():
            report["customer_segment"] = self._default_non_generated_segment()
        if not str(report.get("segment_strategy", "")).strip():
            report["segment_strategy"] = self._default_non_generated_strategy()

        if not report.get("supporting_sources"):
            report["supporting_sources"] = self._format_supporting_sources(state)

        report["agent_metadata"] = {
            "workflow": [
                "sanitize_step",
                "analyze_step",
                "retrieve_step",
                "draft_step",
                "review_step",
                "safety_step",
                "finalize_step",
            ],
            "llm_enabled": bool(self.groq_api_key and ChatGroq is not None),
            "llm_only_mode": True,
            "llm_model": self.llm_model,
            "generation_warnings": state.get("warnings", []),
            "rag_documents_used": [
                doc.get("title", "")
                for doc in state.get("retrieved_practices", [])[:4]
            ],
        }

        return {"final_report": report}

    def _assess_data_quality(self, profile: Dict[str, Any]) -> List[str]:
        notes: List[str] = []

        required_fields = [
            "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
            "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
            "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
            "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
            "MonthlyCharges", "TotalCharges",
        ]

        missing = [
            field
            for field in required_fields
            if field not in profile or str(profile.get(field, "")).strip() == ""
        ]
        if missing:
            notes.append(
                f"Missing fields were imputed: {', '.join(missing[:6])}{'...' if len(missing) > 6 else ''}"
            )

        tenure = _to_float(profile.get("tenure", None), default=-1)
        monthly_charges = _to_float(profile.get("MonthlyCharges", None), default=-1)
        total_charges = _to_float(profile.get("TotalCharges", None), default=-1)

        if tenure < 0 or tenure > 120:
            notes.append("Tenure contained out-of-range values and was clipped to a valid interval.")
        if monthly_charges < 0 or monthly_charges > 400:
            notes.append("Monthly charges contained out-of-range values and were normalized.")
        if total_charges < 0:
            notes.append("Total charges contained invalid negative values and were corrected.")

        return notes

    def _friendly_feature_name(self, feature: str) -> str:
        mapping = {
            "tenure": "Tenure",
            "Contract": "Contract type",
            "MonthlyCharges": "Monthly charges",
            "TotalCharges": "Total charges",
            "PaperlessBilling": "Paperless billing",
            "SeniorCitizen": "Senior citizen status",
            "Dependents": "Dependents",
            "Partner": "Partner status",
            "PhoneService": "Phone service",
            "MultipleLines": "Multiple lines",
            "OnlineSecurity": "Online security",
            "OnlineBackup": "Online backup",
            "DeviceProtection": "Device protection",
            "TechSupport": "Tech support",
            "StreamingTV": "Streaming TV",
            "StreamingMovies": "Streaming movies",
        }
        if feature.startswith("InternetService_"):
            return f"Internet service ({feature.split('InternetService_')[1]})"
        if feature.startswith("PaymentMethod_"):
            return f"Payment method ({feature.split('PaymentMethod_')[1]})"
        return mapping.get(feature, feature)

    def _explain_feature(
        self,
        feature: str,
        contribution: float,
        customer_profile: Dict[str, Any],
    ) -> str:
        direction = "increases" if contribution > 0 else "reduces"
        readable = self._friendly_feature_name(feature)

        if feature.startswith("InternetService_"):
            service = feature.replace("InternetService_", "")
            actual = str(customer_profile.get("InternetService", "unknown"))
            return f"Internet service profile ({actual}) aligned with {service} and {direction} churn risk."

        if feature.startswith("PaymentMethod_"):
            method = feature.replace("PaymentMethod_", "")
            actual = str(customer_profile.get("PaymentMethod", "unknown"))
            return f"Payment method ({actual}) pattern linked to {method} and {direction} churn risk."

        raw_value = customer_profile.get(feature, "n/a")
        return f"{readable} value ({raw_value}) {direction} churn risk in the model."

    def _parse_json_object(self, text: str) -> Dict[str, Any]:
        if not isinstance(text, str):
            return {}

        text = text.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict) and isinstance(parsed.get("report"), dict):
                    return parsed["report"]
                return parsed
            except Exception:
                pass

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}

        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict) and isinstance(parsed.get("report"), dict):
                return parsed["report"]
            return parsed
        except Exception:
            return {}

    def _apply_rule_based_review(self, report: Dict[str, Any], state: RetentionState) -> Dict[str, Any]:
        report = self._ensure_report_schema(report)

        allowed_titles = {source.get("title", "") for source in state.get("supporting_sources", [])}
        cleaned_sources = []
        for source in report.get("supporting_sources", []):
            source_title = source.get("title", "")
            if not allowed_titles or source_title in allowed_titles:
                cleaned_sources.append(
                    {
                        "id": source.get("id", ""),
                        "title": source_title,
                        "url": source.get("url", ""),
                        "justification": source.get(
                            "justification",
                            "Validated against retrieved context.",
                        ),
                    }
                )

        if not cleaned_sources:
            cleaned_sources = self._format_supporting_sources(state)

        report["supporting_sources"] = cleaned_sources

        action_items = []
        for action in report.get("recommended_retention_actions", []):
            rationale = str(action.get("rationale", ""))
            if "guarantee" in rationale.lower():
                action["rationale"] = rationale.replace("guarantee", "aim to improve")
            action_items.append(action)
        report["recommended_retention_actions"] = action_items

        if not report.get("key_contributing_factors"):
            report["key_contributing_factors"] = [
                item.get("explanation", "") for item in state.get("key_factors", [])[:5]
            ]

        if not str(report.get("customer_segment", "")).strip():
            report["customer_segment"] = self._default_non_generated_segment()
        if not str(report.get("segment_strategy", "")).strip():
            report["segment_strategy"] = self._default_non_generated_strategy()

        return report

    def _ensure_report_schema(self, report: Dict[str, Any]) -> Dict[str, Any]:
        report = dict(report or {})
        report.setdefault("churn_risk_summary", "Risk summary unavailable.")
        report.setdefault("customer_segment", "")
        report.setdefault("segment_strategy", "")
        report.setdefault("key_contributing_factors", [])
        report.setdefault("recommended_retention_actions", [])
        report.setdefault("supporting_sources", [])
        report.setdefault("business_and_ethical_disclaimers", [])

        if not isinstance(report["key_contributing_factors"], list):
            report["key_contributing_factors"] = [str(report["key_contributing_factors"])]
        if not isinstance(report["recommended_retention_actions"], list):
            report["recommended_retention_actions"] = []
        if not isinstance(report["supporting_sources"], list):
            report["supporting_sources"] = []
        if not isinstance(report["business_and_ethical_disclaimers"], list):
            report["business_and_ethical_disclaimers"] = []

        return report

    def _representative_profile(self, cluster_df: pd.DataFrame) -> Dict[str, Any]:
        profile: Dict[str, Any] = {}
        candidate_cols = [
            "Contract", "PaymentMethod", "InternetService", "tenure",
            "MonthlyCharges", "PaperlessBilling", "OnlineSecurity", "TechSupport",
        ]

        for col in candidate_cols:
            if col not in cluster_df.columns:
                continue
            series = cluster_df[col]
            if pd.api.types.is_numeric_dtype(series):
                profile[col] = round(float(pd.to_numeric(series, errors="coerce").median()), 2)
            else:
                mode = series.astype(str).mode()
                profile[col] = mode.iloc[0] if not mode.empty else "Unknown"

        return profile

    def _infer_segment_strategy_for_cluster(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        risk_label: str,
        cohort_size: int,
    ) -> Tuple[str, str]:
        llm = self._llm()
        if llm is not None:
            context = {
                "risk_label": risk_label,
                "cohort_size": cohort_size,
                "query": query,
                "retrieved_context": [
                    {
                        "title": doc.get("title", ""),
                        "content": doc.get("content", ""),
                        "source_ids": doc.get("source_ids", []),
                    }
                    for doc in docs
                ],
            }

            system_prompt = (
                "You are a segmentation strategist.\n"
                "Generate one segment label and one concise strategy using retrieved context only.\n"
                "Return strict JSON with keys: segment_label, strategy."
            )

            try:
                response = llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=json.dumps(context)),
                ])
                parsed = self._parse_json_object(response.content)
                segment_label = str(parsed.get("segment_label", "")).strip()
                strategy = str(parsed.get("strategy", "")).strip()
                if segment_label and strategy:
                    return segment_label, strategy
            except Exception:
                pass

        return (
            f"{risk_label} Risk Cohort",
            "No strategy generated in LLM-only mode because valid LLM output was unavailable for this cohort.",
        )

    def generate_retention_report(
        self,
        customer_profile: Dict[str, Any],
        processed_features: Dict[str, float],
        churn_probability: float,
        churn_prediction: int,
    ) -> Dict[str, Any]:
        initial_state: RetentionState = {
            "customer_profile": customer_profile,
            "processed_features": processed_features,
            "churn_probability": float(churn_probability),
            "churn_prediction": int(churn_prediction),
        }

        final_state = self.graph.invoke(initial_state)
        final_report = final_state.get("final_report")
        if isinstance(final_report, dict):
            return final_report

        return self._llm_only_unavailable_report(
            initial_state,
            "LangGraph workflow did not produce a final report object.",
        )

    def generate_batch_segment_strategies(
        self,
        raw_df: pd.DataFrame,
        processed_df: pd.DataFrame,
        churn_probabilities: np.ndarray,
        max_segments: int = 4,
    ) -> Dict[str, Any]:
        output_columns = [
            "Customer Segment",
            "Customers",
            "Average Churn Risk (%)",
            "Key Signals",
            "Recommended Strategy",
            "Evidence Sources",
        ]

        if raw_df is None or processed_df is None or len(processed_df) == 0:
            return {
                "row_segments": [],
                "segment_summary": pd.DataFrame(columns=output_columns),
            }

        probabilities = np.asarray(churn_probabilities, dtype=float).reshape(-1)
        n_rows = len(processed_df)

        if n_rows < 8:
            n_clusters = 1
        else:
            n_clusters = min(max_segments, max(2, int(np.sqrt(n_rows / 3))))

        if n_clusters == 1:
            cluster_ids = np.zeros(n_rows, dtype=int)
        else:
            cluster_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_ids = cluster_model.fit_predict(processed_df.values)

        global_mean = processed_df.mean()
        row_segments = ["Unassigned"] * n_rows
        summary_rows: List[Dict[str, Any]] = []
        seen_labels = set()

        for cluster_id in sorted(np.unique(cluster_ids)):
            indices = np.where(cluster_ids == cluster_id)[0]
            if indices.size == 0:
                continue

            cluster_probs = probabilities[indices]
            mean_prob = float(np.mean(cluster_probs))
            risk_label = _risk_label(mean_prob)

            cluster_processed = processed_df.iloc[indices]
            drift = cluster_processed.mean() - global_mean
            top_features = drift.abs().sort_values(ascending=False).head(4).index.tolist()

            signal_lines: List[str] = []
            for feature in top_features:
                direction = "above" if drift[feature] > 0 else "below"
                signal_lines.append(f"{self._friendly_feature_name(feature)} is {direction} baseline")

            representative_profile = self._representative_profile(raw_df.iloc[indices])
            query = (
                f"Cohort risk: {risk_label}. Cohort size: {int(indices.size)}. "
                f"Representative profile: {json.dumps(representative_profile)}. "
                f"Key signals: {'; '.join(signal_lines)}."
            )

            docs = self._semantic_retrieve(query, top_k=4)
            sources = self._sources_from_docs(docs)
            segment_label, strategy = self._infer_segment_strategy_for_cluster(
                query=query,
                docs=docs,
                risk_label=risk_label,
                cohort_size=int(indices.size),
            )

            base_label = segment_label.strip() or f"{risk_label} Risk RAG Cohort"
            final_label = base_label
            suffix = 2
            while final_label in seen_labels:
                final_label = f"{base_label} ({suffix})"
                suffix += 1
            seen_labels.add(final_label)

            for idx in indices:
                row_segments[int(idx)] = final_label

            source_text = ", ".join([src.get("id", "") for src in sources[:4] if src.get("id")]) or "N/A"

            summary_rows.append(
                {
                    "Customer Segment": final_label,
                    "Customers": int(indices.size),
                    "Average Churn Risk (%)": round(mean_prob * 100, 2),
                    "Key Signals": " | ".join(signal_lines) if signal_lines else "No dominant drift signals",
                    "Recommended Strategy": strategy,
                    "Evidence Sources": source_text,
                }
            )

        segment_summary = pd.DataFrame(summary_rows, columns=output_columns)
        if not segment_summary.empty:
            segment_summary = segment_summary.sort_values(
                "Average Churn Risk (%)",
                ascending=False,
            ).reset_index(drop=True)

        return {
            "row_segments": row_segments,
            "segment_summary": segment_summary,
        }
