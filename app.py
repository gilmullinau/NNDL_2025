import os
import io
import zipfile
from pathlib import Path
import sys

import streamlit as st
import pandas as pd

# Make sure we can import the EDA module
THIS_DIR = Path(__file__).parent.resolve()
sys.path.append(str(THIS_DIR))
import titanic_eda as eda  # type: ignore

st.set_page_config(page_title='Titanic EDA', layout='wide')

st.title('🧭 Titanic EDA Web App')

with st.sidebar:
    st.header('Data source')
    source = st.radio('Choose source:', ['Upload CSV', 'Kaggle (auto-download)'])
    target = st.text_input('Target column', value='survived')
    run_baseline = st.checkbox('Run quick baseline (logreg)', value=False)
    outdir = st.text_input('Output folder (created if missing)', value='eda_report')
    if source == 'Kaggle (auto-download)':
        kaggle_dir = st.text_input('Kaggle data dir', value='kaggle_data')
        st.caption('Requires Kaggle credentials (kaggle.json). See instructions below.')

# Load data
df = None
status_placeholder = st.empty()
if source == 'Upload CSV':
    uploaded = st.file_uploader('Upload a CSV file', type=['csv'])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            status_placeholder.success('CSV loaded.')
        except Exception as e:
            status_placeholder.error(f'Failed to read CSV: {e}')
else:
    if st.button('⬇️ Download Titanic from Kaggle now'):
        try:
            train_path = eda.kaggle_download_titanic(Path(kaggle_dir))
            df = pd.read_csv(train_path)
            status_placeholder.success(f'Downloaded and loaded: {train_path}')
        except Exception as e:
            status_placeholder.error(f'Kaggle download failed: {e}')

# Run EDA
if df is not None:
    st.subheader('Data preview')
    st.dataframe(df.head(50))

    if st.button('▶️ Run EDA'):
        out_path = Path(outdir)
        try:
            df_fe = eda.run_full_eda(df, target=target, outdir=out_path, run_baseline=run_baseline)
            st.success(f'EDA complete. Artifacts saved to: {out_path.resolve()}')

            # Show summary
            summary_md = (out_path / 'SUMMARY.md')
            if summary_md.exists():
                st.markdown(summary_md.read_text())

            # Show images
            pngs = sorted(out_path.glob('*.png'))
            for p in pngs:
                st.image(str(p), caption=p.name, use_column_width=True)

            # Offer ZIP download
            mem = io.BytesIO()
            with zipfile.ZipFile(mem, 'w', zipfile.ZIP_DEFLATED) as zf:
                for path in out_path.rglob('*'):
                    if path.is_file():
                        zf.write(path, arcname=str(path.relative_to(out_path)))
            mem.seek(0)
            st.download_button('📦 Download EDA report (ZIP)', data=mem, file_name='eda_report.zip')

        except Exception as e:
            st.error(f'EDA failed: {e}')

else:
    st.info('Select a data source and load a dataset to begin.')

st.divider()