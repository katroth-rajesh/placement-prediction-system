
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Placement Prediction System", layout="wide")

model = joblib.load("placement_model.pkl")

st.title("🎓 Student Success Analytics & Placement Prediction System")
st.write("Analyze your profile and get placement predictions, company recommendations, and skill improvement suggestions.")

st.sidebar.header("Student Details")

name = st.sidebar.text_input("Student Name")

cgpa = st.sidebar.slider("CGPA", 5.0, 10.0, 7.0)

internships = st.sidebar.number_input("Internships", 0, 5, 1)

projects = st.sidebar.text_input("Projects (Example: Web App, ML Project)")

certifications = st.sidebar.text_input("Certifications (Example: AWS, Coursera ML)")

aptitude = st.sidebar.slider("Aptitude Score", 40, 100, 70)

communication = st.sidebar.text_input("Communication Skills (Example: Presentation, Teamwork)")

technical = st.sidebar.text_input("Technical Skills (Example: Python, Java, SQL)")

backlogs = st.sidebar.number_input("Backlogs", 0, 10, 0)

predict = st.sidebar.button("Predict Placement")

# ---------------- STORE RESULT ----------------
if predict:

    project_score = len(projects.split(",")) if projects else 0
    cert_score = len(certifications.split(",")) if certifications else 0

    tech_score = 0
    tech_lower = technical.lower()

    if "python" in tech_lower:
        tech_score += 2
    if "java" in tech_lower:
        tech_score += 2
    if "sql" in tech_lower:
        tech_score += 1
    if "ml" in tech_lower or "machine learning" in tech_lower:
        tech_score += 2
    if "web" in tech_lower:
        tech_score += 1

    comm_score = len(communication.split(",")) if communication else 0

    input_data = np.array([[cgpa, internships, project_score, cert_score,
                            aptitude, comm_score, tech_score, backlogs]])

    probability = model.predict_proba(input_data)[0][1] * 100

    st.session_state["probability"] = probability

# ---------------- DISPLAY RESULT ----------------
if "probability" in st.session_state:

    probability = st.session_state["probability"]

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📊 Placement Prediction")

        st.metric("Placement Probability", f"{probability:.2f}%")

        if probability > 85:
            tier = "Tier 1"
        elif probability > 70:
            tier = "Tier 2"
        elif probability > 50:
            tier = "Tier 3"
        else:
            tier = "Not Placed"

        st.success(f"{name} falls under: {tier}")

    with col2:

        st.subheader("📈 Probability Chart")

        tier1 = probability * 0.4
        tier2 = probability * 0.35
        tier3 = 100 - (tier1 + tier2)

        labels = ["Tier 1", "Tier 2", "Tier 3"]
        values = [tier1, tier2, tier3]

        fig, ax = plt.subplots()
        ax.bar(labels, values)
        ax.set_ylabel("Probability (%)")
        ax.set_ylim(0, 100)

        st.pyplot(fig)

    st.subheader("🏢 Recommended Companies")

    if tier == "Tier 1":
        companies = ["Google", "Microsoft", "Amazon", "Adobe", "Atlassian"]

    elif tier == "Tier 2":
        companies = ["TCS", "Cognizant", "Accenture", "Capgemini", "Infosys"]

    elif tier == "Tier 3":
        companies = ["Wipro", "Tech Mahindra", "HCL", "Mindtree", "L&T Infotech"]

    else:
        companies = []

    for c in companies:
        st.success(c)

    st.subheader("📌 Skill Improvement Suggestions")

    if tier == "Tier 1":

        suggestions = [
            "Maintain performance",
            "Practice coding problems daily",
            "Continue improving technical skills"
        ]

    elif tier == "Tier 2":

        suggestions = [
            "Improve Data Structures & Algorithms",
            "Build more real-world projects",
            "Improve problem solving to reach Tier 1"
        ]

    elif tier == "Tier 3":

        suggestions = [
            "Improve core technical skills",
            "Work on communication and aptitude",
            "Gain internships and certifications"
        ]

    else:

        suggestions = [
            "Focus on fundamentals",
            "Work on projects",
            "Practice aptitude and coding regularly"
        ]

    for s in suggestions:
        st.warning(s)

    st.subheader("📊 Dataset Insights")

    df = pd.read_csv("placement_dataset.csv")

    col3, col4 = st.columns(2)

    with col3:

        st.write("CGPA Distribution")

        fig1, ax1 = plt.subplots()
        ax1.hist(df["CGPA"], bins=10)

        st.pyplot(fig1)

    with col4:

        st.write("Placement Count")

        fig2, ax2 = plt.subplots()
        df["PlacementStatus"].value_counts().plot(kind="bar", ax=ax2)

        st.pyplot(fig2)

    # ---------------- PRACTICE PROBLEMS ----------------

    st.subheader("🧠 Practice Problems")

    if st.button("Get Practice Problems"):

        st.write("### 📌 Recommended Practice Tasks & Resources")

        if probability > 85:

            st.write("✔ Solve advanced DSA problems (Dynamic Programming, Graphs)")
            st.markdown("🔗 [LeetCode Hard Problems](https://leetcode.com/problemset/all/)")

            st.write("✔ Learn system design concepts")
            st.markdown("🔗 [System Design Tutorial](https://www.geeksforgeeks.org/system-design-tutorial/)")

            st.write("✔ Participate in coding contests")
            st.markdown("🔗 [Codeforces Contests](https://codeforces.com/contests)")

        elif probability > 70:

            st.write("✔ Solve medium-level DSA problems")
            st.markdown("🔗 [LeetCode Medium Problems](https://leetcode.com/problemset/all/)")

            st.write("✔ Practice SQL queries")
            st.markdown("🔗 [HackerRank SQL](https://www.hackerrank.com/domains/sql)")

            st.write("✔ Build 2-3 real-world projects")
            st.markdown("🔗 [Project Ideas](https://www.geeksforgeeks.org/top-project-ideas-for-beginners/)")

        elif probability > 50:

            st.write("✔ Start with basic DSA problems")
            st.markdown("🔗 [DSA Basics](https://www.geeksforgeeks.org/data-structures/)")

            st.write("✔ Practice aptitude daily")
            st.markdown("🔗 [Aptitude Questions](https://www.indiabix.com/aptitude/questions-and-answers/)")

            st.write("✔ Improve coding practice")
            st.markdown("🔗 [HackerRank Practice](https://www.hackerrank.com/domains/tutorials/10-days-of-javascript)")

        else:

            st.write("✔ Learn programming fundamentals")
            st.markdown("🔗 [Python Basics](https://www.w3schools.com/python/)")

            st.write("✔ Practice beginner coding problems")
            st.markdown("🔗 [Beginner Coding](https://www.hackerrank.com/domains/tutorials/10-days-of-python)")

            st.write("✔ Improve aptitude and logical thinking")
            st.markdown("🔗 [Aptitude Basics](https://www.indiabix.com/aptitude/)")

        st.success("💡 Click the links above to start practicing!")
