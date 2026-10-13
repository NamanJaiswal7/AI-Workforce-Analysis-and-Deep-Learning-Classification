import streamlit as st
import time

st.set_page_config(
    page_title="Workforce Analysis Tool",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Workforce Analysis Tool")
st.subheader("Study Program to ISCO-08 Profession Mapping")

# Mock prediction function for the prototype
def predict(text):
    # This is a mock function replacing the actual model inference
    # for the purpose of the prototype UI demonstration.
    time.sleep(1) # simulate processing time
    # Dummy predictions for illustration
    return [
        ("2512", "Software developers", 0.92),
        ("2511", "Systems analysts", 0.85),
        ("2513", "Web and multimedia developers", 0.68),
        ("2521", "Database designers and administrators", 0.45),
        ("2514", "Applications programmers", 0.32)
    ]

# Input section
st.markdown("### Input Study Program Description")
input_text = st.text_area(
    "Paste a study program description (Lithuanian or English):",
    height=200,
    placeholder="Pvz.: Informacinių sistemų programų inžinerijos "
                "studijų programa skirta parengti specialistus, "
                "gebančius projektuoti..."
)

if st.button("🔍 Analyze", type="primary"):
    if input_text.strip():
        with st.spinner("Processing..."):
            # Tokenize and predict
            predictions = predict(input_text)
        
        st.markdown("### Predicted ISCO-08 Professions")
        
        for i, (code, title, score) in enumerate(predictions[:5]):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{code} — {title}**")
                st.progress(score)
            with col2:
                st.metric("Confidence", f"{score:.1%}")
        
        # Human review section
        st.markdown("### Human Review")
        st.info("As required by the EU AI Act (Article 14), "
                "please review and confirm the predictions above.")
        
        review = st.radio(
            "Do you accept these predictions?",
            ["Accept all", "Accept with modifications", "Reject"]
        )
