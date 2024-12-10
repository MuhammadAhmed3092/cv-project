import streamlit as st
from PIL import Image
import io
from model import generate_bleu_score_and_report  # Import function from model.py

# Function to display the Streamlit app
def display_app():
    st.set_page_config(page_title="IU X-Ray Analysis", layout="centered")

    # Header section
    st.markdown("""
        <div style="text-align: center; background-color: #264653; padding: 20px; border-radius: 15px;">
            <h1 style="color: #E9C46A; font-family: Arial, sans-serif;">IU X-Ray Analysis</h1>
            <p style="color: #F4A261;">Upload an X-ray image to provide analyzed report and BLEU score.</p>
        </div>
    """, unsafe_allow_html=True)

    # Upload image section
    uploaded_file = st.file_uploader("Upload an X-ray image (JPG/PNG)", type=["jpg", "png"])

    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-Ray Image", use_column_width=False, width=400)

        # Placeholder for results
        with st.spinner("Processing image..."):
            # Set reference reports for BLEU calculation (dummy reference used here)
            reference_reports = [["The cardiac silhouette and mediastinum size are within normal limits."]]

            # Fetch the result from the model
            bleu_score, report = generate_bleu_score_and_report(uploaded_file, reference_reports)

        # Display BLEU score
        st.write('BLEU Score : ' ,bleu_score)

        # Display the report
        st.markdown("""
            <div style="text-align: center; background-color: #F4A261; padding: 15px; border-radius: 10px; margin-top: 20px;">
                <h3 style="color: #264653;">Generated Report</h3>
            </div>
        """, unsafe_allow_html=True)
        st.text(report)
    else:
        st.markdown("""
            <div style="text-align: center; padding: 20px; background-color: #E63946; border-radius: 10px;">
                <h4 style="color: white;">Please upload an X-ray image to proceed.</h4>
            </div>
        """, unsafe_allow_html=True)

# Run the Streamlit app
if __name__ == "__main__":
    display_app()
