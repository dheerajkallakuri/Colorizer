import numpy as np
import cv2
import streamlit as st
from PIL import Image, ImageOps
import os
import requests
from io import BytesIO

def colorizer(img):
    
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
  
    prototxt = os.path.join("colorization_deploy_v2.prototxt")
    points = os.path.join("pts_in_hull.npy")
    model = os.path.join("colorization_release_v2.caffemodel")
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    pts = np.load(points)
    # add the cluster centers as 1x1 convolutions to the model
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    # scale the pixel intensities to the range [0, 1], and then convert the image from the BGR to Lab color space
    scaled = img.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)
    # resize the Lab image to 224x224 (the dimensions the colorization
    #network accepts), split channels, extract the 'L' channel, and then perform mean centering
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50
    # pass the L channel through the network which will *predict* the 'a' and 'b' channel values
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    # resize the predicted 'ab' volume to the same dimensions as our input image
    ab = cv2.resize(ab, (img.shape[1], img.shape[0]))
    # grab the 'L' channel from the *original* input image (not the
    # resized one) and concatenate the original 'L' channel with the predicted 'ab' channels
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    # convert the output image from the Lab color space to RGB, then clip any values that fall outside the range [0, 1]
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
    colorized = np.clip(colorized, 0, 1)
   
    colorized = (255 * colorized).astype("uint8")
    # Return the colorized images
    return colorized


st.set_page_config(page_title="Colorizer",
                    page_icon="🎨",
                    layout="wide")
st.markdown("# Color your B&W Image")


    
# st.write("""# Color your Black and white image""")

st.write("App to turn colorize your B&W images.")


file = st.sidebar.file_uploader("Please upload an image file", type=["jpg", "png"])
download_button_placeholder = st.sidebar.empty()
download_enabled = False  # Flag to control button state
buffered = None  # Placeholder for image data

if st.button("Load a random B&W image"):
    try:
        # API call to fetch a random black-and-white image (for example, using Unsplash)
        response = requests.get('https://unsplash.it/400/400?grayscale')
        image = Image.open(BytesIO(response.content))
        st.text("Random B&W image loaded")
        img = np.array(image)
    except Exception as e:
        st.text(f"Error loading image: {e}")
        img = None
else:
    if file is None:
        st.text("You haven't uploaded an image file")
        img = None
    else:
        image = Image.open(file)
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

if img is not None:
    st.text("Your original image")
    st.image(image, use_column_width=True)

    st.text("Your colorized image")
    color = colorizer(img)
    
    st.image(color, use_column_width=True)

    colorized_pil = Image.fromarray(color)

    buffered = BytesIO()
    colorized_pil.save(buffered, format="PNG")
    buffered.seek(0)

    # Enable the download button after the colorized image is generated
    download_enabled = True

# Sidebar: Display download button (enabled after image is processed)
if download_enabled:
    download_button_placeholder.download_button(
        label="Download Colorized Image",
        data=buffered,
        file_name="colorized_image.png",
        mime="image/png"
    )
else:
    # Display a disabled button before the image is generated
    download_button_placeholder.button(label="Download Colorized Image", disabled=True)

print("done!")




