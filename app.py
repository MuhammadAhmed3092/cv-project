import streamlit as st
from PIL import Image
import io
from model import generate_bleu_score_and_report  # Import function from model.py

# Function to display the Streamlit app
def display_app():
    st.set_page_config(page_title="IU X-Ray Analysis", layout="wide")

    # Header section
    st.markdown("""
        <div style="text-align: center; background-color: #1E3A8A; padding: 30px 15px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
            <h1 style="color: white; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-size: 36px;">IU X-Ray Analysis</h1>
            <p style="color: #f8f9fa; font-size: 18px; font-family: Arial, sans-serif;">Upload an X-ray image to analyze the BLEU score and generate a detailed report.</p>
        </div>
    """, unsafe_allow_html=True)

    # Upload image section
    uploaded_file = st.file_uploader("Upload an X-ray image (JPG/PNG)", type=["jpg", "png"], label_visibility="collapsed")

    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-Ray Image", use_column_width=True)

        # Placeholder for results
        with st.spinner("Processing image..."):
            # Set reference reports for BLEU calculation (dummy reference used here)
            reference_reports = [["The cardiac silhouette and mediastinum size are within normal limits."]]
            
            # Fetch the result from the model
            bleu_score, report = generate_bleu_score_and_report(uploaded_file, reference_reports)

        # Display BLEU score in a stylish card
        st.markdown("""
            <div style="text-align: center; margin-top: 30px; padding: 20px; background-color: #48C9B0; border-radius: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                <h3 style="color: #1D4F5D; font-size: 28px;">BLEU Score</h3>
                <p style="font-size: 24px; color: white; font-weight: bold;">{}</p>
            </div>
        """.format(bleu_score), unsafe_allow_html=True)

        # Display the report in a nice section
        st.markdown("<h3 style='text-align: center; margin-top: 40px; color: #1E3A8A;'>Generated Report</h3>", unsafe_allow_html=True)
        st.text(report)

        # Allow downloading of the report in a stylish button
        report_bytes = io.BytesIO(report.encode())
        st.download_button(
            label="Download Report",
            data=report_bytes,
            file_name="xray_report.txt",
            mime="text/plain",
            use_container_width=True
        )

    else:
        st.markdown("""
            <div style="text-align: center; padding: 40px; background-color: #A7C7E7; border-radius: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                <h4 style="color: #023e8a; font-family: Arial, sans-serif;">Please upload an X-ray image to proceed.</h4>
            </div>
        """, unsafe_allow_html=True)

# Run the Streamlit app
if __name__ == "__main__":
    display_app()
