from pathlib import Path
from io import BytesIO
import hashlib
import json
import numpy as np
from PIL import Image
import streamlit as st
from model import load_checkpoint
from inference import predict_pair

st.set_page_config(page_title='Building Change Explorer', page_icon='🛰️', layout='wide')
st.title('Building Change Explorer')
st.write('Compare two dates of the same place to highlight predicted building changes.')
st.caption('Upload aligned RGB images at the same scale. This model does not register images or detect forest loss. Best suited to imagery similar to LEVIR-CD (0.5 m/pixel).')

@st.cache_resource
def get_model(path, modified):
    return load_checkpoint(path, 'cpu')

def png(array):
    out = BytesIO()
    Image.fromarray(array).save(out, format='PNG')
    return out.getvalue()

import tempfile
import urllib.request

MODEL_URL = (
    "https://github.com/RoyalAskarov/satellite-change-detection/"
    "releases/download/v1.0.0/levir_cd_best.pt"
)

# Fingerprint of your exported model, checked from your local file.
MODEL_SHA256 = (
    "69e7c93bfa89a81f73f73c39f1ead761"
    "5911b651b968f1e67b803b742276fe85"
)

checkpoint_path = Path(__file__).parent / "levir_cd_best.pt"

try:
    if not checkpoint_path.exists():
        temporary_path = None

        with st.spinner("Downloading the trained model. Please wait…"):
            try:
                request = urllib.request.Request(
                    MODEL_URL,
                    headers={"User-Agent": "BuildingChangeExplorer/1.0"},
                )
                digest = hashlib.sha256()

                with urllib.request.urlopen(request, timeout=60) as response:
                    with tempfile.NamedTemporaryFile(
                        dir=checkpoint_path.parent,
                        suffix=".download",
                        delete=False,
                    ) as temporary:
                        temporary_path = Path(temporary.name)

                        while chunk := response.read(1024 * 1024):
                            temporary.write(chunk)
                            digest.update(chunk)

                if digest.hexdigest() != MODEL_SHA256:
                    raise ValueError(
                        "Model download verification failed. Please retry."
                    )

                temporary_path.replace(checkpoint_path)

            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)

    model, metadata = get_model(
        str(checkpoint_path),
        checkpoint_path.stat().st_mtime_ns,
    )

except Exception as exc:
    st.error(f"Cannot load the trained model: {exc}")
    st.info("Check the connection and refresh the page to retry.")
    st.stop()

left,right = st.columns(2)
before_file = left.file_uploader('Before image', type=['png','jpg','jpeg'])
after_file = right.file_uploader('After image', type=['png','jpg','jpeg'])
threshold = st.sidebar.slider('Change threshold', 0.05, 0.95, float(metadata.get('threshold',0.5)), 0.05)
st.sidebar.caption('Lower thresholds mark more pixels as changed. The reported test metrics use 0.50.')
st.sidebar.write('Model: Siamese ResNet-18 U-Net')
st.sidebar.write('Training data: LEVIR-CD')
if before_file is None or after_file is None:
    st.stop()
try:
    a = Image.open(before_file)
    b = Image.open(after_file)
    if a.size != b.size:
        raise ValueError('Images have different dimensions. Align them before uploading.')
    if a.width*a.height > 4_194_304:
        raise ValueError('For this demo use at most 4 megapixels. Crop both dates to the same area.')
    a,b = a.convert('RGB'),b.convert('RGB')
except Exception as exc:
    st.error(f'Cannot use these images: {exc}')
    st.stop()
left.image(a, caption='Before', use_container_width=True)
right.image(b, caption='After', use_container_width=True)
key = hashlib.sha256(before_file.getvalue()+after_file.getvalue()+str(checkpoint_path.stat().st_mtime_ns).encode()).hexdigest()
if st.button('Detect building changes', type='primary'):
    bar = st.progress(0.0, text='Comparing image patches…')
    with st.spinner('Running the model…'):
        probability = predict_pair(model,a,b,tile=int(metadata['tile_size']),progress=bar.progress)
    st.session_state['prediction'] = (key, probability)
    bar.empty()
result = st.session_state.get('prediction')
if result is not None and result[0] == key:
    probability = result[1]
    mask = probability >= threshold
    overlay = np.array(b).copy()
    overlay[mask] = (0.55*overlay[mask]+0.45*np.array([255,50,65])).astype(np.uint8)
    c1,c2 = st.columns(2)
    c1.image(mask.astype(np.uint8)*255, caption='Predicted change mask', use_container_width=True)
    c2.image(overlay, caption='Predicted changes in red', use_container_width=True)
    st.metric('Pixels predicted as building change', f'{mask.mean()*100:.2f}%')
    st.caption('This is a pixel fraction, not a building count or a measured ground area. Change includes additions and removals without distinguishing them. Model probabilities are not calibrated confidence scores.')
    st.download_button('Download mask', png(mask.astype(np.uint8)*255), 'change_mask.png','image/png')
    st.download_button('Download overlay', png(overlay),'change_overlay.png','image/png')
    st.download_button('Download probability image', png((probability*65535).round().astype(np.uint16)), 'change_probability_16bit.png','image/png')
    report = {'threshold':threshold, 'changed_pixels':int(mask.sum()), 'total_pixels':int(mask.size),
              'changed_percent':float(mask.mean()*100), 'width':a.width,'height':a.height,
              'architecture':metadata['architecture'], 'dataset':'LEVIR-CD'}
    st.download_button('Download summary', json.dumps(report,indent=2), 'summary.json','application/json')
