import io
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import streamlit as st
from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

load_dotenv()
ROLE_LIBRARY: Dict[str, Dict[str, List[str]]] = {
    "data scientist": {
        "core": [
            "python",
            "pandas",
            "numpy",
            "statistics",
            "machine learning",
            "sql",
            "feature engineering",
        ],
        "nice_to_have": ["pytorch", "tensorflow", "mlops", "experiment tracking"],
    },
    "data engineer": {
        "core": [
            "python",
            "sql",
            "data modeling",
            "etl",
            "airflow",
            "spark",
            "cloud storage",
        ],
        "nice_to_have": ["kafka", "dbt", "orchestration", "observability"],
    },
    "software engineer": {
        "core": [
            "python",
            "java",
            "oop",
            "testing",
            "system design",
            "rest apis",
            "git",
        ],
        "nice_to_have": ["docker", "kubernetes", "ci/cd", "observability"],
    },
    "product manager": {
        "core": [
            "roadmapping",
            "prioritization",
            "analytics",
            "stakeholder management",
            "experimentation",
        ],
        "nice_to_have": ["sql", "a/b testing", "story writing", "design thinking"],
    },
}


def _to_set(raw: str) -> List[str]:
    parts = [p.strip().lower() for p in raw.replace(";", ",").split(",") if p.strip()]
    return parts


def _join_lines(items: List[str]) -> str:
    return "\n".join(f"- {line}" for line in items)


@tool
def skills_gap_analyzer(
    current_skills: str,
    target_role: str,
    target_requirements: str = "",
    years_experience: float = 0.0,
    desired_level: str = "mid-level",
) -> str:
    """Compare user skills to a target role and propose a learning plan."""
    target_key = target_role.lower().strip()
    parsed_current = set(_to_set(current_skills))
    parsed_target = set(_to_set(target_requirements)) if target_requirements else set()
    library = ROLE_LIBRARY.get(target_key, {})

    if not parsed_target:
        parsed_target.update(library.get("core", []))
        parsed_target.update(library.get("nice_to_have", []))

    missing = sorted(parsed_target - parsed_current)
    covered = sorted(parsed_target & parsed_current)
    extras = sorted(parsed_current - parsed_target)

    pace = "aggressive" if desired_level.lower().startswith("sen") else "steady"
    learning_plan = [
        f"Prioritize missing core items first; target a {pace} 10-week plan.",
        "Pair each week with one project that proves the skill (GitHub/portfolio).",
        "Block 2x weekly mock interviews or whiteboard sessions with peers.",
    ]
    if years_experience < 2:
        learning_plan.append(
            "Emphasize fundamentals: data structures, SQL fluency, and version control."
        )
    else:
        learning_plan.append(
            "Add architecture/system design drills to demonstrate senior judgement."
        )

    sections = [
        f"Target role: {target_role or 'unspecified'} | Desired level: {desired_level}",
        f"Skills you already cover ({len(covered)}): {', '.join(covered) or 'none yet'}",
        f"Skill gaps to close ({len(missing)}): {', '.join(missing) or 'none'}",
        f"Adjacent strengths (not requested but useful): {', '.join(extras) or 'n/a'}",
        "Suggested 4-week sprint themes:",
        _join_lines(
            [
                "Week 1: Fundamentals + refresh key libraries/tools used in target teams.",
                "Week 2: Build a small end-to-end project aligned to the role.",
                "Week 3: Deepen one specialization (ML model, data pipeline, or API).",
                "Week 4: Polish storytelling, metrics, and documentation.",
            ]
        ),
        "Learning plan:",
        _join_lines(learning_plan),
    ]
    return "\n".join(sections)


@tool
def resume_scorer(
    resume_text: str,
    target_role: str = "",
    job_description: str = "",
) -> str:
    """Score a resume out of 10 with actionable feedback."""
    text = resume_text.lower()
    word_count = len(resume_text.split())
    metrics_hits = sum(ch.isdigit() for ch in resume_text)
    keywords = set(_to_set(job_description)) if job_description else set()

    matched_keywords = [kw for kw in keywords if kw in text]
    score = 6.0
    tips: List[str] = []

    if metrics_hits >= 6:
        score += 1.5
    elif metrics_hits >= 2:
        score += 1.0
    else:
        tips.append("Add metrics (latency, revenue, accuracy, coverage) to each bullet.")

    if 325 <= word_count <= 750:
        score += 1.0
    else:
        tips.append("Keep the resume concise (roughly 1 page, 325-750 words).")

    if matched_keywords:
        score += min(2.0, 0.2 * len(matched_keywords))
    else:
        tips.append("Mirror 6-8 keywords from the target job description.")

    if "education" not in text:
        tips.append("Add an Education section with degree, school, and graduation year.")
    if "project" not in text:
        tips.append("Include 1-2 role-aligned projects with outcomes.")
    if "python" in text or "sql" in text or "java" in text:
        score += 0.5

    score = max(0.0, min(10.0, round(score, 1)))
    summary = (
        f"Resume score: {score}/10 for {target_role or 'your target role'}\n"
        f"Matched keywords ({len(matched_keywords)}): "
        f"{', '.join(matched_keywords) or 'none'}\n"
        "Top fixes:\n"
        f"{_join_lines(tips) if tips else '- Strong balance of impact, keywords, and structure.'}"
    )
    return summary


@tool
def salary_estimator(
    job_title: str,
    location: str,
    years_experience: float,
    seniority: str = "",
    industry: str = "tech",
) -> str:
    """Estimate a realistic US salary band based on role, location, and experience."""
    title_key = job_title.lower().strip()
    base_ranges: Dict[str, Tuple[int, int]] = {
        "software engineer": (95000, 160000),
        "data scientist": (100000, 170000),
        "data engineer": (105000, 175000),
        "ml engineer": (115000, 185000),
        "product manager": (110000, 180000),
    }
    low, high = base_ranges.get(title_key, (90000, 150000))

    exp_factor = 1 + max(-0.3, min(0.6, (years_experience - 3) * 0.045))
    seniority_map = {
        "junior": 0.85,
        "mid": 1.0,
        "senior": 1.2,
        "staff": 1.35,
        "principal": 1.55,
    }
    seniority_factor = seniority_map.get(seniority.lower(), 1.0)

    loc = location.lower()
    location_map = {
        "san francisco": 1.35,
        "bay area": 1.32,
        "new york": 1.25,
        "seattle": 1.18,
        "austin": 1.1,
        "boston": 1.15,
        "remote": 1.05,
    }
    location_factor = 1.0
    for key, factor in location_map.items():
        if key in loc:
            location_factor = factor
            break

    industry_factor = 1.0 if industry.lower() in {"tech", "software"} else 0.92
    final_low = int(low * exp_factor * seniority_factor * location_factor * industry_factor)
    final_high = int(high * exp_factor * seniority_factor * location_factor * industry_factor)

    details = [
        f"Base range reference ({title_key or 'role'}): ${low:,} - ${high:,}",
        f"Adjusted for {years_experience} yrs exp: x{exp_factor:.2f}",
        f"Seniority factor ({seniority or 'unspecified'}): x{seniority_factor:.2f}",
        f"Location factor ({location or 'unknown'}): x{location_factor:.2f}",
        f"Industry factor ({industry}): x{industry_factor:.2f}",
        f"Estimated range: ${final_low:,} - ${final_high:,} total comp (salary-only; equity/bonus varies).",
    ]
    return "\n".join(details)


@tool
def interview_question_generator(
    role: str,
    difficulty: str = "mixed",
    focus: str = "balanced",
    count: int = 6,
) -> str:
    """Generate technical and behavioral interview questions for the given role."""
    role_key = role.lower().strip()
    difficulty = difficulty.lower()
    focus = focus.lower()

    technical_bank = [
        "Walk through a recent system or project design; what trade-offs did you make?",
        "Explain how you would profile and speed up a slow data pipeline.",
        "How do you design logging, tracing, and alerting for a new service?",
        "Describe a time you simplified a complex system without losing capability.",
        "How would you estimate capacity for peak traffic and plan scaling?",
        "What tests and checks would you add before deploying a risky change?",
    ]
    behavioral_bank = [
        "Tell me about a disagreement with a teammate—what did you do?",
        "Describe a failure; what changed in your approach afterward?",
        "How do you choose between moving fast and building quality?",
        "When have you influenced direction without authority?",
        "How do you onboard quickly to an unfamiliar codebase or domain?",
    ]
    role_specific = {
        "data scientist": [
            "How would you set success metrics for an ML model in production?",
            "Explain a feature you engineered that materially improved model lift.",
            "What are your steps for handling data leakage and concept drift?",
        ],
        "data engineer": [
            "Design a reliable backfill strategy for a broken daily pipeline.",
            "How do you choose storage formats (parquet/csv/json) and partitioning?",
            "Explain idempotency in ETL jobs and how to enforce it.",
        ],
        "ml engineer": [
            "How do you monitor model quality post-deployment without labels?",
            "What is your rollout plan for a new model version with uncertainty?",
            "Discuss the trade-offs of online vs batch feature computation.",
        ],
        "software engineer": [
            "Design an API that needs strict backward compatibility.",
            "How do you prevent and mitigate cascading failures in distributed systems?",
            "Explain how you would debug intermittent production errors.",
        ],
        "product manager": [
            "Write a brief PRD outline for a feature that improves retention.",
            "How do you validate problem vs solution before committing resources?",
            "Give an example of using data to de-risk a launch.",
        ],
    }

    chosen: List[str] = []
    if focus in {"technical", "tech"}:
        chosen.extend(technical_bank)
    elif focus in {"behavioral", "people"}:
        chosen.extend(behavioral_bank)
    else:
        chosen.extend(technical_bank[:3] + behavioral_bank[:3])

    chosen.extend(role_specific.get(role_key, [])[:3])
    difficulty_note = f"Difficulty: {difficulty}. Focus: {focus}."
    trimmed = chosen[: max(3, min(count, len(chosen)))]
    lines = [difficulty_note, "Questions:"] + [
        f"{idx+1}. {q}" for idx, q in enumerate(trimmed)
    ]
    return "\n".join(lines)


BASE_SYSTEM_PROMPT = """
You are a concise, pragmatic career counseling agent with access to specialized tools
(skills_gap_analyzer, resume_scorer, salary_estimator, interview_question_generator).
Use tools when a query maps to skills analysis, resume scoring, salary estimation,
or interview preparation. Keep answers concrete and prioritized.
Always summarize next steps and keep the response scannable.
""".strip()


def build_system_prompt(default_role: str, location: str, years: float) -> str:
    return (
        BASE_SYSTEM_PROMPT
        + f"\n\nUser defaults:\n"
        f"- Target role: {default_role or 'unspecified'}\n"
        f"- Location: {location or 'unspecified'}\n"
        f"- Years experience: {years}\n"
        "Use these as context when the user is vague."
    )


@dataclass
class AgentConfig:
    model: str
    temperature: float
    api_key: str
    default_role: str
    default_location: str
    default_years: float


@st.cache_resource(show_spinner=False)
def get_llm(model: str, temperature: float, api_key: str) -> ChatGoogleGenerativeAI:
    if not api_key:
        raise ValueError(
            "Google API key is required. Set GOOGLE_API_KEY env var or fill the sidebar field."
        )
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=api_key,
        convert_system_message_to_human=True,
    )


def ensure_history() -> InMemoryChatMessageHistory:
    if "message_history" not in st.session_state:
        st.session_state.message_history = InMemoryChatMessageHistory()
    return st.session_state.message_history


def run_agent(user_input: str, config: AgentConfig) -> Tuple[str, List[Tuple[str, str]]]:
    history = ensure_history()
    llm = get_llm(config.model, config.temperature, config.api_key)
    tools = [skills_gap_analyzer, resume_scorer, salary_estimator, interview_question_generator]
    tool_map = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    history.add_message(HumanMessage(content=user_input))
    system_message = SystemMessage(
        content=build_system_prompt(
            config.default_role, config.default_location, config.default_years
        )
    )

    messages = [system_message, *history.messages]
    ai_message: AIMessage = llm_with_tools.invoke(messages)
    history.add_message(ai_message)

    tool_traces: List[Tuple[str, str]] = []
    while ai_message.tool_calls:
        for call in ai_message.tool_calls:
            tool = tool_map.get(call["name"])
            if not tool:
                result_text = f"Requested tool {call['name']} is unavailable."
            else:
                try:
                    result = tool.invoke(call["args"])
                    result_text = (
                        json.dumps(result, indent=2)
                        if isinstance(result, dict)
                        else str(result)
                    )
                except Exception as exc:
                    result_text = f"Tool execution error: {exc}"
            tool_traces.append((call["name"], result_text))
            history.add_message(ToolMessage(content=result_text, tool_call_id=call["id"]))

        messages = [system_message, *history.messages]
        ai_message = llm_with_tools.invoke(messages)
        history.add_message(ai_message)

    return ai_message.content, tool_traces


def extract_text_from_upload(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    if uploaded_file.type.startswith("text/"):
        return uploaded_file.read().decode("utf-8", errors="ignore")
    if PdfReader is None:
        raise RuntimeError(
            "PyPDF2 is required for PDF parsing. Install with `pip install PyPDF2`."
        )
    reader = PdfReader(uploaded_file)
    pages_text = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages_text)


def scrape_job_description(url: str) -> str:
    if not url:
        return ""
    try:
        import requests
        from bs4 import BeautifulSoup

        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        text = " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))
        return text[:15000]
    except Exception:
        return ""


def extract_skills_with_llm(llm: ChatGoogleGenerativeAI, text: str, max_skills: int = 25) -> List[str]:
    if not text.strip():
        return []
    prompt = (
        "From the following resume or job description text, extract up to "
        f"{max_skills} distinct skills as a comma-separated list. "
        "Only output the comma-separated skills, nothing else.\n\n"
        f"TEXT:\n{text}"
    )
    result = llm.invoke(prompt)
    raw = result.content if isinstance(result.content, str) else "".join(
        ch["text"] for ch in result.content if isinstance(ch, dict) and "text" in ch
    )
    return _to_set(raw)

st.set_page_config(
    page_title="Career Care",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.header("Configuration")

model = st.sidebar.selectbox(
    "Model",
    ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"],
    index=0,
)
default_role = st.sidebar.text_input("Default target role", value="Data Scientist")
default_location = st.sidebar.text_input(
    "Default location", value="San Francisco, CA"
)
default_years = st.sidebar.number_input(
    "Default years of experience", min_value=0.0, max_value=40.0, value=3.0, step=0.5
)
remote_friendly = st.sidebar.checkbox("Remote-friendly", value=True)

st.sidebar.markdown("---")
temperature = st.sidebar.slider("Creativity (temperature)", 0.0, 1.2, 0.25, 0.05)
default_key = os.getenv("GOOGLE_API_KEY", "")
api_key = st.sidebar.text_input(
    "Google API Key",
    value=default_key,
    type="password",
    help="You can also set GOOGLE_API_KEY in your environment.",
)

show_tools = st.sidebar.checkbox("Show tool call details in Chat tab", value=False)

if st.sidebar.button("Update chat agent"):
    st.sidebar.success("New configuration will be used for future chat messages.")

if st.sidebar.button("Start fresh chat"):
    st.session_state.pop("message_history", None)
    st.session_state.pop("chat_log", None)
    st.sidebar.info("Chat history cleared.")

st.sidebar.markdown("---")
st.sidebar.subheader("Quick prompts")
st.sidebar.markdown(
    """
- Skills gap tab: Upload resume + job link, then analyze.
- Salary tab: "Estimate salary for remote ML Engineer in Seattle with 5 years."
- Interview tab: "Hard PM questions about product sense and metrics."
""".strip()
)

llm_for_tools: ChatGoogleGenerativeAI | None = None
if api_key:
    llm_for_tools = get_llm(model, temperature, api_key)

st.title("Career Care")
tabs = st.tabs(["Skills Gap", "Resume Scorer", "Salary", "Interview", "Chat"])
with tabs[0]:
    st.subheader("Skills Gap Analyzer")
    st.caption(
        "Upload your resume and a job description or link. "
        "We’ll detect skills, compare them, and suggest a learning path."
    )

    col1, col2 = st.columns(2)
    with col1:
        resume_file = st.file_uploader(
            "Upload resume (PDF or text)", type=["pdf", "txt", "md"]
        )
        override_skills = st.text_area(
            "Add/override skills (optional)",
            placeholder="Python, SQL, Tableau, ML, ...",
        )
    with col2:
        job_link = st.text_input("Job link to scrape (optional)")
        jd_text = st.text_area(
            "Or paste the job description",
            placeholder="Paste job description here...",
            height=160,
        )

    target_role = st.text_input("Target role", value=default_role)
    desired_level = st.selectbox(
        "Desired level", ["junior", "mid-level", "senior"], index=1
    )

    if st.button("Analyze gap", type="primary"):
        if not llm_for_tools:
            st.error("Please provide a valid Google API key in the sidebar.")
        else:
            try:
                resume_text = ""
                if resume_file is not None:
                    resume_text = extract_text_from_upload(resume_file)

                if not jd_text and job_link:
                    scraped = scrape_job_description(job_link)
                    if scraped:
                        jd_text = scraped
                    else:
                        st.warning(
                            "Could not scrape the job link. You can paste the JD text instead."
                        )

                detected_resume_skills = extract_skills_with_llm(
                    llm_for_tools, resume_text
                )
                if override_skills.strip():
                    detected_resume_skills.extend(_to_set(override_skills))

                detected_jd_skills = extract_skills_with_llm(llm_for_tools, jd_text)

                current_skills_str = ", ".join(sorted(set(detected_resume_skills)))
                jd_skills_str = ", ".join(sorted(set(detected_jd_skills)))

                result = skills_gap_analyzer.invoke(
                    {
                        "current_skills": current_skills_str,
                        "target_role": target_role,
                        "target_requirements": jd_skills_str,
                        "years_experience": float(default_years),
                        "desired_level": desired_level,
                    }
                )

                st.success("Skills gap analysis")
                st.text(result)

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Detected skills in resume**")
                    st.write(", ".join(sorted(set(detected_resume_skills))) or "None")
                with c2:
                    st.markdown("**Detected skills in job description**")
                    st.write(", ".join(sorted(set(detected_jd_skills))) or "None")

                if jd_text:
                    with st.expander("Preview of scraped / pasted job description"):
                        st.write(jd_text[:6000])
            except Exception as exc:
                st.error(f"Error during analysis: {exc}")

with tabs[1]:
    st.subheader("Resume Reviewer")
    st.caption(
        "Upload your resume to get a 0–10 score, strengths, and targeted suggestions."
    )

    col1, col2 = st.columns(2)
    with col1:
        resume_file_2 = st.file_uploader(
            "Upload resume (PDF or text)", type=["pdf", "txt", "md"], key="resume2"
        )
    with col2:
        pasted_resume = st.text_area(
            "Or paste resume text",
            placeholder="Paste your resume here if you don't want to upload a file...",
            height=200,
        )

    jd_for_resume = st.text_area(
        "Job description (optional)",
        placeholder="Paste job description here for keyword matching...",
        height=160,
    )

    if st.button("Analyze Resume", type="primary"):
        if not llm_for_tools:
            st.error("Please provide a valid Google API key in the sidebar.")
        else:
            try:
                resume_text_2 = ""
                if resume_file_2 is not None:
                    resume_text_2 = extract_text_from_upload(resume_file_2)
                elif pasted_resume.strip():
                    resume_text_2 = pasted_resume
                else:
                    st.error("Please upload a resume or paste the text.")
                    st.stop()

                summary = resume_scorer.invoke(
                    {
                        "resume_text": resume_text_2,
                        "target_role": default_role,
                        "job_description": jd_for_resume,
                    }
                )

                first_line = summary.splitlines()[0]
                score_value = "?"
                if "Resume score:" in first_line:
                    try:
                        score_value = first_line.split("Resume score:")[1].split(
                            "/10"
                        )[0].strip()
                    except Exception:
                        pass

                col_score, col_text = st.columns([1, 3])
                with col_score:
                    st.markdown(f"## {score_value}/10")
                    st.markdown("**Overall rating**")
                with col_text:
                    st.markdown(summary)

            except Exception as exc:
                st.error(f"Error while scoring resume: {exc}")

with tabs[2]:
    st.subheader("Salary Estimator")
    st.caption("Estimate realistic salary ranges based on role, location, and YOE.")

    job_title = st.text_input("Job title", value=default_role)
    loc = st.text_input("Location", value=default_location)
    years = st.number_input(
        "Years of experience", min_value=0.0, max_value=40.0, value=default_years, step=0.5
    )
    seniority = st.selectbox(
        "Seniority", ["", "junior", "mid", "senior", "staff", "principal"], index=2
    )
    industry = st.text_input("Industry", value="tech")

    if st.button("Estimate salary", type="primary"):
        result = salary_estimator.invoke(
            {
                "job_title": job_title,
                "location": loc if not remote_friendly else f"{loc} (remote-friendly)",
                "years_experience": float(years),
                "seniority": seniority or "mid",
                "industry": industry,
            }
        )
        st.success("Estimated compensation")
        st.text(result)
with tabs[3]:
    st.subheader("Interview Question Generator")
    st.caption(
        "Generate technical and behavioral questions for your next interview prep session."
    )

    role_for_q = st.text_input("Role", value=default_role)
    difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard", "mixed"], index=3)
    tech_count = st.slider("Number of technical questions", 1, 10, 5)
    beh_count = st.slider("Number of behavioral questions", 1, 10, 3)

    if st.button("Generate questions", type="primary"):
        tech_block = interview_question_generator.invoke(
            {
                "role": role_for_q,
                "difficulty": difficulty,
                "focus": "technical",
                "count": tech_count,
            }
        )
        beh_block = interview_question_generator.invoke(
            {
                "role": role_for_q,
                "difficulty": difficulty,
                "focus": "behavioral",
                "count": beh_count,
            }
        )

        st.success("Interview questions")
        st.markdown("**Technical questions**")
        st.text(tech_block)
        st.markdown("**Behavioral questions**")
        st.text(beh_block)

with tabs[4]:
    st.subheader("Chat")
    st.caption(
        "Ask free-form questions. The agent will pick tools automatically and remember context."
    )

    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []

    def render_history():
        for msg in st.session_state.chat_log:
            st.chat_message(msg["role"]).markdown(msg["content"])

    render_history()

    user_prompt = st.chat_input(
        "Ask about skills gaps, resume feedback, salary, or interview prep..."
    )
    if user_prompt:
        st.session_state.chat_log.append({"role": "user", "content": user_prompt})
        st.chat_message("user").markdown(user_prompt)

        try:
            with st.spinner("Thinking..."):
                reply, traces = run_agent(
                    user_prompt,
                    AgentConfig(
                        model=model,
                        temperature=temperature,
                        api_key=api_key,
                        default_role=default_role,
                        default_location=default_location,
                        default_years=default_years,
                    ),
                )
        except Exception as exc:
            reply = (
                "Error while contacting Gemini. Check that your Google API key is set and the model name is valid.\n"
                f"Details: {exc}"
            )
            traces = []

        st.session_state.chat_log.append({"role": "assistant", "content": reply})
        st.chat_message("assistant").markdown(reply)

        if show_tools and traces:
            with st.expander("Tool calls + outputs", expanded=False):
                for name, output in traces:
                    st.markdown(f"**{name}**")
                    st.code(output)
