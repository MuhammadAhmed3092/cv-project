import streamlit as st
from PIL import Image
import io
from model import generate_bleu_score_and_report  # Import function from model.py

# Function to display the Streamlit app
def display_app():
    st.set_page_config(page_title="IU X-Ray Analysis", layout="centered")

    # Header section
    st.markdown("""
        <style>
            .header-container {
                text-align: center;
                background: #264653;
                padding: 40px 20px;
                border-radius: 15px;
                box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
            }
            .header-title {
                color: #E9C46A;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                font-size: 36px;
                margin-bottom: 10px;
            }
            .header-subtitle {
                color: #F4A261;
                font-size: 18px;
                font-family: Arial, sans-serif;
            }
        </style>
        <div class="header-container">
            <h1 class="header-title">IU X-Ray Analysis</h1>
            <p class="header-subtitle">Upload an X-ray image to analyze the BLEU score and generate a detailed report.</p>
        </div>
    """, unsafe_allow_html=True)

    # Upload image section
    uploaded_file = st.file_uploader("Upload an X-ray image (JPG/PNG)", type=["jpg", "png"], label_visibility="collapsed")

    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        st.markdown("""
            <style>
                .image-container {
                    display: flex;
                    justify-content: center;
                    margin-top: 20px;
                }
                .uploaded-image {
                    border-radius: 15px;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
                }
            </style>
            <div class="image-container">
                <img class="uploaded-image" src="data:image/png;base64,{}" width="50%" />
            </div>
        """.format(image), unsafe_allow_html=True)

        # Placeholder for results
        with st.spinner("Processing image..."):
            # Set reference reports for BLEU calculation (dummy reference used here)
            reference_reports = [["The cardiac silhouette and mediastinum size are within normal limits."]]

            # Fetch the result from the model
            bleu_score, report = generate_bleu_score_and_report(uploaded_file, reference_reports)

        # Display BLEU score in a stylish card
        st.markdown(f"""
            <style>
                .bleu-card {{
                    text-align: center;
                    margin: 30px auto;
                    padding: 20px;
                    background: #2A9D8F;
                    border-radius: 15px;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
                }}
                .bleu-title {{
                    color: #F4F4F4;
                    font-size: 24px;
                    font-weight: bold;
                }}
                .bleu-score {{
                    color: #E76F51;
                    font-size: 32px;
                    font-weight: bold;
                    margin-top: 10px;
                }}
            </style>
            <div class="bleu-card">
                <div class="bleu-title">BLEU Score</div>
                <div class="bleu-score">{bleu_score}</div>
            </div>
        """, unsafe_allow_html=True)

        # Display the report in a responsive layout
        st.markdown("""
            <style>
                .report-container {
                    margin: 40px auto;
                    padding: 20px;
                    background: #F4A261;
                    border-radius: 10px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                }
                .report-title {
                    text-align: center;
                    color: #264653;
                    font-size: 24px;
                    font-weight: bold;
                }
                .report-content {
                    margin-top: 10px;
                    color: #1D3557;
                    font-family: monospace;
                    font-size: 16px;
                    line-height: 1.6;
                }
            </style>
            <div class="report-container">
                <div class="report-title">Generated Report</div>
                <div class="report-content">{}</div>
            </div>
        """.format(report), unsafe_allow_html=True)

        # Allow downloading of the report
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
            <style>
                .upload-prompt {
                    text-align: center;
                    padding: 40px;
                    background: #E63946;
                    color: white;
                    border-radius: 15px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                }
                .upload-prompt h4 {
                    font-family: Arial, sans-serif;
                    font-size: 18px;
                    margin: 0;
                }
            </style>
            <div class="upload-prompt">
                <h4>Please upload an X-ray image to proceed.</h4>
            </div>
        """, unsafe_allow_html=True)

# Run the Streamlit app
if __name__ == "__main__":
    display_app()
