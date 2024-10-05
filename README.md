# Colorizer App - Black & White to Color Image Conversion

Welcome to the **Colorizer App**, a deep learning-powered tool designed to convert black-and-white images into vibrant color images. This application leverages the state-of-the-art **Colorful Image Colorization** technique developed by **Zhang et al.** at UC Berkeley, presented in their 2016 ECCV paper: "Colorful Image Colorization."

Try the Colorizer app live: [Colorizer App Live Demo](https://colorizer.streamlit.app/)

## Overview of the Model

The Colorizer app is built on the powerful **Convolutional Neural Network (CNN)** model from Zhang et al.'s work, which tackled the challenge of colorizing black-and-white images. Here's a summary of the approach:

### Problem Background
- Earlier methods for colorizing grayscale images relied heavily on **manual human annotation**, leading to results that often appeared desaturated and lacked realism.
- Zhang et al. revolutionized the process by using a **CNN** to automatically "hallucinate" the colors that grayscale images would have if they were taken in full color.

<img width="1103" alt="Screenshot 2024-10-05 at 2 09 42 PM" src="https://github.com/user-attachments/assets/9b1e5ddf-e383-480f-a050-55b3ba0f42f2">


### Training Process
The model was trained on over a million color images from the **ImageNet** dataset. During training:
1. The images were first converted from the RGB color space to the **Lab color space**, which consists of three channels:
   - **L channel**: Represents the lightness or intensity.
   - **a channel**: Represents the green-red color spectrum.
   - **b channel**: Represents the blue-yellow color spectrum.
   
2. The **L channel** (grayscale) was used as the input to the CNN, and the network was trained to predict the **a** and **b** channels (color information).

3. Once the network predicted the a and b channels, they were combined with the L channel, and the image was converted back to RGB format for the final colorized output.

### Key Techniques
- **Class Rebalancing**: A special class-rebalancing technique was applied to encourage the network to produce more diverse and vibrant colorizations during training.
- **Mean Annealing**: This technique helps ensure more plausible colorizations by modifying how the predictions are averaged over time.

Using these techniques, the model can produce realistic and vivid colorizations of black-and-white photos.

## Features of the Colorizer App

This app provides a simple, intuitive interface where users can:
- **Upload their own black-and-white images** for colorization.
- **Load random black-and-white images** via the **Unsplash API** for colorization.
- **Download the colorized images** directly from the app.

The app was built using **Streamlit**, a popular framework for building web applications with Python. 

## How It Works
1. Upload a black-and-white image, or load a sample using the Unsplash API.
2. The app uses the Zhang et al. colorization model to process the image.
3. Once the image is colorized, you can download it directly from the app.

## User Interface

<img width="1205" alt="Screenshot 2024-10-05 at 2 07 12 PM" src="https://github.com/user-attachments/assets/9afad572-23d3-402a-b1dd-73f08eb9d8c4">

## Results

<table>
  <tr>
    <th>B&W Image</th>
    <th>Colorized Imaged</th>
  </tr>
  <tr>
    <td><img width="420" alt="Screenshot 2024-10-05 at 2 06 37 PM" src="https://github.com/user-attachments/assets/109f1391-8e67-4bcc-a98e-2c4ca222065f"></td>
    <td><img width="420" alt="Screenshot 2024-10-05 at 2 06 47 PM" src="https://github.com/user-attachments/assets/6bf7c45a-8ea2-45c9-917b-93fb572754e2"></td>
  </tr>
</table>


# References

1. [Colorful Image Colorization (ECCV 2016)](https://richzhang.github.io/colorization/).
2. [Black and white image colorization with OpenCV and Deep Learning](https://pyimagesearch.com/2019/02/25/black-and-white-image-colorization-with-opencv-and-deep-learning/).
3. [Inspiration for colorizer app](https://github.com/dhananjayan-r/Colorizer).


