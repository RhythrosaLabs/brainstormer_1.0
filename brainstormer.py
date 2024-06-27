
# I M P O R T S

# Standard Library
import os
import io
import warnings

# Third Party
from PIL import Image, ImageDraw, ImageFont, ImageTk, ImageOps, ImageEnhance, ImageFilter
from PIL import ImageStat
from PIL import ImageGrab
import tkinter as tk
from tkinter import ttk
from stability_sdk import client
from stability_sdk import api
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from stability_sdk.animation import AnimationArgs, Animator
from tqdm import tqdm
import cv2
import numpy as np
import random
import openai
from midiutil import MIDIFile
from pydub import AudioSegment
from pydub.generators import Sine
import time
import requests
from tkinter import filedialog
from tkinter import colorchooser
from tkinter import messagebox
import io
import datetime

# My Library
import words
from words import sample_words


#======================================================================================================================
#======================================================================================================================

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

#======================================================================================================================
#======================================================================================================================


# A P I    K E Y S

# Set your OpenAI API key
openai.api_key = 'PUT YOUR OPENAI API KEY HERE'

# Set up environment variables
os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'
os.environ['STABILITY_KEY'] = 'PUT YOUR STABILITY AI API KEY HERE'

# Set up connection to the API
stability_api = client.StabilityInference(
    key=os.environ['STABILITY_KEY'],
    verbose=True,
    engine="stable-diffusion-xl-beta-v2-2-2",
    upscale_engine="stable-diffusion-x4-latent-upscaler",  # Use x4 upscaler
)

STABILITY_HOST = "grpc.stability.ai:443"
STABILITY_KEY = "PUT YOUR STABILITY AI API KEY HERE" # Your API key from dreamstudio.ai

api_context = api.Context(STABILITY_HOST, STABILITY_KEY)


#======================================================================================================================
#======================================================================================================================


# S A V E    C H A T B O T

# Function to save a specific conversation to a text file
def save_specific_conversation(tab_index):
    # Ensure the directory exists
    if not os.path.exists('conversations'):
        os.makedirs('conversations')

    conversation = conversations[tab_index]
    # Use current datetime to create a unique filename
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(f'conversations/conversation_{tab_index}_{timestamp_str}.json', 'w') as f:
        # Use json.dump to write the conversation to the file
        json.dump(conversation, f)



#======================================================================================================================
#======================================================================================================================

# B U T T O N   C L I C K S

# RANDOMIZER

def button_randomize_click():
    text_entry.delete(0, tk.END)
    random_sentence = words.generate_random_sentence()
    text_entry.insert(0, random_sentence)


#======================================================================================================================
#======================================================================================================================


# D A L L E    T E X T    T O    I M A G E

def generate_image_dalle():
    prompt = text_entry.get()

    # Directly use the "Create" mode
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="512x512"
    )

    image_url = response['data'][0]['url']

    # Download the image from the URL
    img_data = requests.get(image_url).content
    img = Image.open(io.BytesIO(img_data))

    # Save the image
    image_filename = f"{prompt}.png"
    img.save(image_filename)

    # Display the image in the GUI
    img.thumbnail((256, 256))  # Resize the image to fit the GUI
    img_tk = ImageTk.PhotoImage(img)
    lbl.config(image=img_tk)
    lbl.image = img_tk



#======================================================================================================================
#======================================================================================================================

# D A L L E   V A R I A T I O N S 


# Create a function to generate variations of the image using DALL·E
def generate_variations():
    try:
        global img_for_itf  # Ensure that img_for_itf variable is global
        image = img_for_itf  # Use the uploaded image

        # Convert image to bytes
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")

        response = openai.Image.create_variation(
            image=image_bytes.getvalue(),
            n=1,  # Number of variations to generate
            size="1024x1024",  # Size of the generated images
            response_format="url"  # Format of the generated images
        )

        generated_image_url = response['data'][0]['url']
        response = requests.get(generated_image_url)
        generated_image = Image.open(io.BytesIO(response.content))

        # Resize the image
        resized_image = generated_image.resize((256, 256), Image.ANTIALIAS)

        # Save and display the generated image
        variation_filename = "{}_variation.png".format(timestamp)
        resized_image.save(variation_filename)
        img_tk = ImageTk.PhotoImage(resized_image)
        fx_image_label.config(image=img_tk)
        fx_image_label.image = img_tk

        result_label.config(text="Variations generated successfully!")
    except Exception as e:
        result_label.config(text="Error: " + str(e))


#======================================================================================================================
#======================================================================================================================


# S T A B L E    D I F F U S I O N    T E X T    T O    I M A G E

def text_to_seed(text):
    return sum([ord(c) for c in text])


def generate_image_diffusion(prompt, seed, steps):
    answers = stability_api.generate(
        prompt=prompt,
        seed=seed,
        steps=steps,
        cfg_scale=8.0,
        width=512,
        height=512,
        samples=1,
        sampler=generation.SAMPLER_K_DPMPP_2M
    )
    return answers

def generate_image_and_save():
    input_text = text_entry.get()
    seed = text_to_seed(input_text)

    # Get the number of steps from the steps_entry field
    try:
        steps = int(steps_entry.get())
    except ValueError:
        print("Invalid steps value. Please enter a valid integer.")
        return

    # Pass the steps value to the generate_image_diffusion function
    answers = generate_image_diffusion(input_text, seed, steps)

    for resp in answers:
        for artifact in resp.artifacts:
            if artifact.finish_reason == generation.FILTER:
                warnings.warn(
                    "Your request activated the API's safety filters and could not be processed."
                    "Please modify the prompt and try again."
                )
            if artifact.type == generation.ARTIFACT_IMAGE:
                img = Image.open(io.BytesIO(artifact.binary))
                image_filename = f"{input_text}.png"
                img.save(image_filename)

                # Display the image in the GUI
                img.thumbnail((256, 256))  # Resize the image to fit the GUI
                img_tk = ImageTk.PhotoImage(img)
                lbl.config(image=img_tk)
                lbl.image = img_tk

#======================================================================================================================
#======================================================================================================================

# G P T    S M A R T    C H A O S

def random_chaos_single():
    # Generate 1 sentence with ChatGPT
    messages = [
        {"role": "system", "content": "You will act as if you are a creative screenwriter."},
        {"role": "user", "content": "Invent a movie scene in 1 sentence that is 15 words or less:"}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        max_tokens=100,
        temperature=1
    )
    sentence = response['choices'][0]['message']['content'].strip()

    # Insert the sentence into the specific text field
    text_entry.delete(0, tk.END)
    text_entry.insert(0, sentence)

#======================================================================================================================
#======================================================================================================================


# I M A G E   T O    I M A G E    F U N C T I O N S

def upload_image_for_itf():
    global img_for_itf  # Declare global variable to store the uploaded image for image-to-image function
    global current_img  # Declare global variable to store the current image
    file_path = filedialog.askopenfilename()
    img_for_itf = Image.open(file_path)
    img_for_itf.thumbnail((256, 256))  # Adjust the thumbnail size to be multiples of 64
    img_tk = ImageTk.PhotoImage(img_for_itf)
    lbl.configure(image=img_tk)
    lbl.image = img_tk
    current_img = img_for_itf  # Update the current image

#----------------------------------------------------------------------------------------------------------------------

# image to image parameters

def image_to_image():
    global current_img  # Ensure that the current_img variable is global

    # Retrieve user input from the Image-to-Image prompt entry widget
    image_to_image_prompt = image_to_image_entry.get()

    # Use the current image as the input image
    img = current_img

    answers = stability_api.generate(
            prompt=image_to_image_prompt,  # Use the user's prompt
            init_image=img,
            start_schedule=0.6,
            seed=123467458,
            steps=30,
            cfg_scale=8.0,
            width=1024,
            height=1024,
            sampler=generation.SAMPLER_K_DPMPP_2M
        )
    for resp in answers:
            for artifact in resp.artifacts:
                if artifact.finish_reason == generation.FILTER:
                    warnings.warn(
                        "Your request activated the API's safety filters and could not be processed."
                        "Please modify the prompt and try again.")
                if artifact.type == generation.ARTIFACT_IMAGE:
                    img = Image.open(io.BytesIO(artifact.binary))
                    img.thumbnail((256, 256))  # Resize the image to fit the GUI
                    img_tk = ImageTk.PhotoImage(img)
                    lbl.config(image=img_tk)
                    lbl.image = img_tk
                    img.save(f"./{image_to_image_prompt}_img2img.png")  # Save the image in the same directory as the script
                    current_img = img  # Update the current image

    # Assuming the image_label is the label we want to display the image in
    image_label.grid(row=2, column=3)


#======================================================================================================================
#======================================================================================================================


# F X    A N D    I M A G E    E D I T I N G


# M I R R O R    I M A G E

def mirror_image():
    try:
        input_text = text_entry.get()
        image_filename = f"{input_text}.png"
        img = Image.open(image_filename)

        # Mirror the image
        img_mirrored = ImageOps.mirror(img)

        # Save and display the mirrored image
        mirrored_filename = f"{input_text}_mirrored.png"
        img_mirrored.save(mirrored_filename)
        img_mirrored.thumbnail((256, 256))  # Resize the image to fit the GUI
        img_mirrored_tk = ImageTk.PhotoImage(img_mirrored)
        fx_image_label.config(image=img_mirrored_tk)
        fx_image_label.image = img_mirrored_tk

    except Exception as e:
        print("An error occurred during the image mirroring:", e)

#----------------------------------------------------------------------------------------------------------------------

# I N V E R T    C O L O R S

def invert_colors():
    try:
        input_text = text_entry.get()
        seed = text_to_seed(input_text)
        image_filename = f"{input_text}.png"
        img = Image.open(image_filename)

        # Invert the colors of the image
        img_inverted = ImageOps.invert(img)

        # Save and display the inverted image
        inverted_filename = f"{input_text}_inverted.png"
        img_inverted.save(inverted_filename)
        img_inverted.thumbnail((256, 256))  # Resize the image to fit the GUI
        img_inverted_tk = ImageTk.PhotoImage(img_inverted)
        fx_image_label.config(image=img_inverted_tk)
        fx_image_label.image = img_inverted_tk

    except Exception as e:
        print("An error occurred during the color inversion:", e)

#----------------------------------------------------------------------------------------------------------------------

# P I X E L A T E

def pixelate_image():
    input_text = text_entry.get()
    seed = text_to_seed(input_text)
    image_filename = f"{input_text}.png"
    img = Image.open(image_filename)

    # Pixelate the image
    pixel_size = 5  # Define the size of the pixels
    img_pixelated = img.resize(
        (img.size[0] // pixel_size, img.size[1] // pixel_size),
        Image.NEAREST
    ).resize(img.size, Image.NEAREST)

    # Save and display the pixelated image
    pixelated_filename = f"{input_text}_pixelated.png"
    img_pixelated.save(pixelated_filename)
    img_pixelated.thumbnail((256, 256))  # Resize the image to fit the GUI
    img_pixelated_tk = ImageTk.PhotoImage(img_pixelated)
    fx_image_label.config(image=img_pixelated_tk)
    fx_image_label.image = img_pixelated_tk


#----------------------------------------------------------------------------------------------------------------------


# R E M O V E   B A C K G R O U N D

def remove_background():
    try:
        input_text = text_entry.get()
        image_filename = f"{input_text}.png"
        img = Image.open(image_filename)

        # Convert the image to OpenCV format (BGR)
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Define mask, bgdModel, fgdModel
        mask = np.zeros(cv_img.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        # Define rectangle for GrabCut
        rect = (50, 50, cv_img.shape[1]-100, cv_img.shape[0]-100)
        # Apply GrabCut
        cv2.grabCut(cv_img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

        # Create binary mask where foreground is 1 and background is 0
        binary_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        # Multiply original image with binary mask to get result
        result = cv_img * binary_mask[:, :, np.newaxis]

        # Convert the result back to PIL format (RGB)
        img_removed_bg = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

        # Save and display the image with the background removed
        removed_bg_filename = f"{input_text}_removed_bg.png"
        img_removed_bg.save(removed_bg_filename)
        img_removed_bg.thumbnail((256, 256))  # Resize the image to fit the GUI
        img_removed_bg_tk = ImageTk.PhotoImage(img_removed_bg)
        fx_image_label.config(image=img_removed_bg_tk)
        fx_image_label.image = img_removed_bg_tk
    except Exception as e:
        print("An error occurred while removing the background:", e)


#----------------------------------------------------------------------------------------------------------------------

# G R E Y S C A L E

def greyscale_image():
    try:
        input_text = text_entry.get()
        seed = text_to_seed(input_text)
        image_filename = f"{input_text}.png"
        img = Image.open(image_filename)

        # Convert the image to greyscale
        img_greyscale = img.convert('L')

        # Save and display the greyscale image
        greyscale_filename = f"{input_text}_greyscale.png"
        img_greyscale.save(greyscale_filename)
        img_greyscale.thumbnail((256, 256))  # Resize the image to fit the GUI
        img_greyscale_tk = ImageTk.PhotoImage(img_greyscale)
        fx_image_label.config(image=img_greyscale_tk)
        fx_image_label.image = img_greyscale_tk

    except Exception as e:
        print("An error occurred during the greyscale conversion:", e)

#----------------------------------------------------------------------------------------------------------------------

# S A T U R A T E

def enhance_saturation():
    try:
        input_text = text_entry.get()
        image_filename = f"{input_text}.png"
        img = Image.open(image_filename)

        # Enhance the saturation of the image
        enhancer = ImageEnhance.Color(img)
        enhanced_img = enhancer.enhance(1.5)  # Increase the saturation

        # Save and display the enhanced image
        enhanced_filename = f"{input_text}_saturate.png"
        enhanced_img.save(enhanced_filename)
        enhanced_img.thumbnail((256, 256))  # Resize the image to fit the GUI
        enhanced_img_tk = ImageTk.PhotoImage(enhanced_img)
        fx_image_label.config(image=enhanced_img_tk)
        fx_image_label.image = enhanced_img_tk

    except Exception as e:
        print("An error occurred during enhancing saturation:", e)

#----------------------------------------------------------------------------------------------------------------------

# S E P I A

def sepia_tone():
    try:
        input_text = text_entry.get()
        image_filename = f"{input_text}.png"
        img = Image.open(image_filename)

        # Convert the image to sepia tone
        img_sepia = img.convert("L").convert("RGB")
        sepia_data = img_sepia.load()
        width, height = img_sepia.size
        for y in range(height):
            for x in range(width):
                r, g, b = img_sepia.getpixel((x, y))
                sepia_red = int(r * 0.393 + g * 0.769 + b * 0.189)
                sepia_green = int(r * 0.349 + g * 0.686 + b * 0.168)
                sepia_blue = int(r * 0.272 + g * 0.534 + b * 0.131)
                sepia_data[x, y] = (sepia_red, sepia_green, sepia_blue)

        # Save and display the sepia tone image
        sepia_filename = f"{input_text}_sepia.png"
        img_sepia.save(sepia_filename)
        img_sepia.thumbnail((256, 256))  # Resize the image to fit the GUI
        img_sepia_tk = ImageTk.PhotoImage(img_sepia)
        fx_image_label.config(image=img_sepia_tk)
        fx_image_label.image = img_sepia_tk

    except Exception as e:
        print("An error occurred during applying sepia tone:", e)

#----------------------------------------------------------------------------------------------------------------------

# B L U R

def blur_image():
    try:
        input_text = text_entry.get()
        image_filename = f"{input_text}.png"
        img = Image.open(image_filename)

        # Blur the image
        img_blurred = img.filter(ImageFilter.BLUR)

        # Save and display the blurred image
        blurred_filename = f"{input_text}_blur.png"
        img_blurred.save(blurred_filename)
        img_blurred.thumbnail((256, 256))  # Resize the image to fit the GUI
        img_blurred_tk = ImageTk.PhotoImage(img_blurred)
        fx_image_label.config(image=img_blurred_tk)
        fx_image_label.image = img_blurred_tk

    except Exception as e:
        print("An error occurred during image blurring:", e)

#----------------------------------------------------------------------------------------------------------------------

# emboss

def emboss_image():
    try:
        input_text = text_entry.get()
        image_filename = f"{input_text}.png"
        img = Image.open(image_filename)

        # Apply emboss filter
        img_emboss = img.filter(ImageFilter.EMBOSS)

        # Save and display the embossed image
        emboss_filename = f"{input_text}_emboss.png"
        img_emboss.save(emboss_filename)
        img_emboss.thumbnail((256, 256))  # Resize the image to fit the GUI
        img_emboss_tk = ImageTk.PhotoImage(img_emboss)
        fx_image_label.config(image=img_emboss_tk)
        fx_image_label.image = img_emboss_tk

    except Exception as e:
        print("An error occurred during the emboss filter:", e)

#---------------------------------------------------------------------------------------------------------------------

# solarize

def solarize_image():
    try:
        input_text = text_entry.get()
        image_filename = f"{input_text}.png"
        img = Image.open(image_filename)

        # Apply solarize filter
        img_solarize = ImageOps.solarize(img, threshold=128)

        # Save and display the solarized image
        solarize_filename = f"{input_text}_solarize.png"
        img_solarize.save(solarize_filename)
        img_solarize.thumbnail((256, 256))  # Resize the image to fit the GUI
        img_solarize_tk = ImageTk.PhotoImage(img_solarize)
        fx_image_label.config(image=img_solarize_tk)
        fx_image_label.image = img_solarize_tk

    except Exception as e:
        print("An error occurred during the solarize filter:", e)

#----------------------------------------------------------------------------------------------------------------------

# posterize

def posterize_image():
    try:
        input_text = text_entry.get()
        image_filename = f"{input_text}.png"
        img = Image.open(image_filename)

        # Apply posterize filter
        img_posterize = ImageOps.posterize(img, bits=3)  # Reduce to 4 bits

        # Save and display the posterized image
        posterize_filename = f"{input_text}_posterize.png"
        img_posterize.save(posterize_filename)
        img_posterize.thumbnail((256, 256))  # Resize the image to fit the GUI
        img_posterize_tk = ImageTk.PhotoImage(img_posterize)
        fx_image_label.config(image=img_posterize_tk)
        fx_image_label.image = img_posterize_tk

    except Exception as e:
        print("An error occurred during the posterize filter:", e)

#----------------------------------------------------------------------------------------------------------------------

# sharpen

def sharpen_image():
    try:
        input_text = text_entry.get()
        image_filename = f"{input_text}.png"
        img = Image.open(image_filename)

        # Apply sharpen filter
        img_sharpen = img.filter(ImageFilter.SHARPEN)

        # Save and display the sharpened image
        sharpen_filename = f"{input_text}_sharpen.png"
        img_sharpen.save(sharpen_filename)
        img_sharpen.thumbnail((256, 256))  # Resize the image to fit the GUI
        img_sharpen_tk = ImageTk.PhotoImage(img_sharpen)
        fx_image_label.config(image=img_sharpen_tk)
        fx_image_label.image = img_sharpen_tk

    except Exception as e:
        print("An error occurred during the sharpen filter:", e)

#----------------------------------------------------------------------------------------------------------------------

# VHS Glitch Effect

def apply_vhs_glitch_effect():
    try:
        input_text = text_entry.get()
        image_filename = f"{input_text}.png"
        img = Image.open(image_filename)

        # Convert the image to a numpy array for manipulation
        img_array = np.array(img)

        # Apply VHS glitch-style effect to the image (more dramatic)
        red_channel = img_array[:, :, 0]
        green_channel = img_array[:, :, 1]
        blue_channel = img_array[:, :, 2]

        shifted_red = np.roll(red_channel, 30, axis=1)
        shifted_green = np.roll(green_channel, -20, axis=0)
        shifted_blue = np.roll(blue_channel, 40, axis=1)

        glitched_img_array = np.stack((shifted_red, shifted_green, shifted_blue), axis=2)
        glitched_img = Image.fromarray(glitched_img_array)

        # Save and display the glitched image
        glitched_filename = f"{input_text}_glitch.png"
        glitched_img.save(glitched_filename)
        glitched_img.thumbnail((256, 256))  # Resize the image to fit the GUI
        glitched_img_tk = ImageTk.PhotoImage(glitched_img)
        fx_image_label.config(image=glitched_img_tk)
        fx_image_label.image = glitched_img_tk

    except Exception as e:
        print("An error occurred during applying VHS glitch effect:", e)

#----------------------------------------------------------------------------------------------------------------------

# oil painting
        
def apply_oil_painting_effect():
    try:
        input_text = text_entry.get()
        image_filename = f"{input_text}.png"
        img = Image.open(image_filename)

        # Apply the oil painting effect
        img = img.filter(ImageFilter.ModeFilter(size=5))  # Increase size for larger "brush strokes"
        img = img.filter(ImageFilter.MaxFilter(size=5))  # Increase size for larger "brush strokes"

        # Define a convolution matrix for edge enhancement
        convolution_matrix = ImageFilter.Kernel(
            size=(3, 3),
            kernel=[-2, -1, 0, -1, 1, 1, 0, 1, 2],
            scale=sum([-2, -1, 0, -1, 1, 1, 0, 1, 2]),
        )
        img = img.filter(convolution_matrix)

        # Apply sharpening filter multiple times
        for _ in range(3):
            img = img.filter(ImageFilter.SHARPEN)

        # Save and display the oil painting effect image
        oil_painting_filename = f"{input_text}_oil_painting.png"
        img.save(oil_painting_filename)
        img.thumbnail((256, 256))  # Resize the image to fit the GUI
        img = ImageTk.PhotoImage(img)
        fx_image_label.config(image=img)
        fx_image_label.image = img

    except Exception as e:
        print("An error occurred during applying oil painting effect:", e)
        
#----------------------------------------------------------------------------------------------------------------------


# Flip Vertically

def flip_vertically():
    try:
        input_text = text_entry.get()
        image_filename = f"{input_text}.png"
        img = Image.open(image_filename)

        # Flip the image vertically
        flipped_img = img.transpose(Image.FLIP_TOP_BOTTOM)

        # Save and display the flipped image
        flipped_filename = f"{input_text}_flipped.png"
        flipped_img.save(flipped_filename)
        flipped_img.thumbnail((256, 256))  # Resize the image to fit the GUI
        flipped_img_tk = ImageTk.PhotoImage(flipped_img)
        fx_image_label.config(image=flipped_img_tk)
        fx_image_label.image = flipped_img_tk

    except Exception as e:
        print("An error occurred during flipping the image vertically:", e)
        
#----------------------------------------------------------------------------------------------------------------------

# Edge Detection

def edge_detection():
    try:
        input_text = text_entry.get()
        image_filename = f"{input_text}.png"
        img = Image.open(image_filename)

        # Apply edge detection filter
        img_edges = img.filter(ImageFilter.FIND_EDGES)

        # Save and display the edge detected image
        edges_filename = f"{input_text}_edges.png"
        img_edges.save(edges_filename)
        img_edges.thumbnail((256, 256))
        img_edges_tk = ImageTk.PhotoImage(img_edges)
        fx_image_label.config(image=img_edges_tk)
        fx_image_label.image = img_edges_tk

    except Exception as e:
        print("An error occurred during edge detection:", e)


#----------------------------------------------------------------------------------------------------------------------

def mosaic():
    try:
        input_text = text_entry.get()
        image_filename = f"{input_text}.png"
        img = cv2.imread(image_filename)

        # Mosaic effect
        block_size = 8  # Smaller blocks for more detail
        img_mosaic = img.copy()
        for x in range(0, img.shape[0], block_size):
            for y in range(0, img.shape[1], block_size):
                img_mosaic[x:x+block_size, y:y+block_size] = np.mean(np.mean(img[x:x+block_size, y:y+block_size], axis=0), axis=0)
        
        # Edge detection
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_edges = cv2.Canny(img_gray, 50, 150)

        # Blend the mosaic and edges
        alpha = 0.7  # Change this to control blending (0 = only mosaic, 1 = only edges)
        img_mosaic = cv2.cvtColor(img_mosaic, cv2.COLOR_BGR2GRAY)
        img_blend = cv2.addWeighted(img_mosaic, alpha, img_edges, 1-alpha, 0)

        # Save and display the blended image
        blend_filename = f"{input_text}_blend.png"
        cv2.imwrite(blend_filename, img_blend)
        img_blend = Image.open(blend_filename)
        img_blend.thumbnail((256, 256))  # Resize the image to fit the GUI
        img_blend_tk = ImageTk.PhotoImage(img_blend)
        fx_image_label.config(image=img_blend_tk)
        fx_image_label.image = img_blend_tk

    except Exception as e:
        print("An error occurred during creative mosaic effect:", e)
        
#----------------------------------------------------------------------------------------------------------------------
        
# Noise Reduction

def noise_reduction():
    try:
        # Use cv2.fastNlMeansDenoisingColored for noise reduction
        input_text = text_entry.get()
        image_filename = f"{input_text}.png"
        img = cv2.imread(image_filename)

        # Apply noise reduction
        img_noise_reduced = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

        # Save and display the noise-reduced image
        noise_reduced_filename = f"{input_text}_noise_reduced.png"
        cv2.imwrite(noise_reduced_filename, img_noise_reduced)
        img_noise_reduced = Image.open(noise_reduced_filename)
        img_noise_reduced.thumbnail((256, 256))  # Resize the image to fit the GUI
        img_noise_reduced_tk = ImageTk.PhotoImage(img_noise_reduced)
        fx_image_label.config(image=img_noise_reduced_tk)
        fx_image_label.image = img_noise_reduced_tk

    except Exception as e:
        print("An error occurred during noise reduction:", e)

#----------------------------------------------------------------------------------------------------------------------

# Pencil Sketch

def pencil_sketch():
    try:
        # Create a pencil sketch by first converting the image to grayscale, and then applying an inversion and a binary threshold
        input_text = text_entry.get()
        image_filename = f"{input_text}.png"
        img = cv2.imread(image_filename)

        # Convert to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply inversion
        img_inverted = cv2.bitwise_not(img_gray)

        # Apply binary threshold
        _, img_pencil_sketch = cv2.threshold(img_inverted, 150, 255, cv2.THRESH_BINARY)

        # Save and display the pencil sketch image
        pencil_sketch_filename = f"{input_text}_pencil_sketch.png"
        cv2.imwrite(pencil_sketch_filename, img_pencil_sketch)
        img_pencil_sketch = Image.open(pencil_sketch_filename)
        img_pencil_sketch.thumbnail((256, 256))  # Resize the image to fit the GUI
        img_pencil_sketch_tk = ImageTk.PhotoImage(img_pencil_sketch)
        fx_image_label.config(image=img_pencil_sketch_tk)
        fx_image_label.image = img_pencil_sketch_tk

    except Exception as e:
        print("An error occurred during pencil sketch effect:", e)

#----------------------------------------------------------------------------------------------------------------------

# Cartoonize

def cartoonize():
    try:
        # Cartoonizing an image involves reducing the color palette and applying bilateral filter to remove noise
        input_text = text_entry.get()
        image_filename = f"{input_text}.png"
        img = cv2.imread(image_filename)

        # Reduce color palette
        img_reduced_color = cv2.pyrMeanShiftFiltering(img, sp=20, sr=60)

        # Convert to grayscale
        img_gray = cv2.cvtColor(img_reduced_color, cv2.COLOR_BGR2GRAY)

        # Apply median blur
        img_blur = cv2.medianBlur(img_gray, 3)

        # Use adaptive threshold to create an edge mask
        img_edge = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2)

        # Combine the color image with the edge mask to create a cartoon effect
        img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2BGR)
        img_cartoonized = cv2.bitwise_and(img_reduced_color, img_edge)

        # Save and display the cartoonized image
        cartoonized_filename = f"{input_text}_cartoonized.png"
        cv2.imwrite(cartoonized_filename, img_cartoonized)
        img_cartoonized = Image.open(cartoonized_filename)
        img_cartoonized.thumbnail((256, 256))
        img_cartoonized_tk = ImageTk.PhotoImage(img_cartoonized)
        fx_image_label.config(image=img_cartoonized_tk)
        fx_image_label.image = img_cartoonized_tk

    except Exception as e:
        print("An error occurred during cartoonize:", e)

#----------------------------------------------------------------------------------------------------------------------

# Watercolor

def watercolor():
    try:
        # Watercolor effect can be achieved by applying a stylization filter
        input_text = text_entry.get()
        image_filename = f"{input_text}.png"
        img = cv2.imread(image_filename)

        # Apply stylization
        img_stylized = cv2.stylization(img, sigma_s=60, sigma_r=0.07)

        # Save and display the stylized image
        watercolor_filename = f"{input_text}_watercolor.png"
        cv2.imwrite(watercolor_filename, img_stylized)
        img_watercolor = Image.open(watercolor_filename)
        img_watercolor.thumbnail((256, 256))
        img_watercolor_tk = ImageTk.PhotoImage(img_watercolor)
        fx_image_label.config(image=img_watercolor_tk)
        fx_image_label.image = img_watercolor_tk

    except Exception as e:
        print("An error occurred during watercolor effect:", e)

#----------------------------------------------------------------------------------------------------------------------

# Vignette Effect

def vignette():
    try:
        input_text = text_entry.get()
        image_filename = f"{input_text}.png"
        img = cv2.imread(image_filename)

        rows, cols = img.shape[:2]
        # Generate vignette mask using Gaussian kernels
        kernel_x = cv2.getGaussianKernel(int(1.5*cols),200)
        kernel_y = cv2.getGaussianKernel(int(1.5*rows),200)
        kernel = kernel_y @ kernel_x.T
        mask = kernel[int(0.5*rows):, int(0.5*cols):]
        mask = cv2.resize(mask, (cols, rows))
        mask = mask / mask.max()
        img_vignette = np.copy(img)
        
        # Apply the mask to each channel in the input image
        for i in range(3):
            img_vignette[:,:,i] = img_vignette[:,:,i] * mask

        # Save and display the vignetted image
        vignette_filename = f"{input_text}_vignette.png"
        cv2.imwrite(vignette_filename, img_vignette)
        img_vignette = Image.open(vignette_filename)
        img_vignette.thumbnail((256, 256))  # Resize the image to fit the GUI
        img_vignette_tk = ImageTk.PhotoImage(img_vignette)
        fx_image_label.config(image=img_vignette_tk)
        fx_image_label.image = img_vignette_tk

    except Exception as e:
        print("An error occurred during vignette effect:", e)

#----------------------------------------------------------------------------------------------------------------------

#vintage

def vintage_filter():
    try:
        input_text = text_entry.get()
        image_filename = f"{input_text}.png"
        img = cv2.imread(image_filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert to sepia
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
        img_sepia = cv2.transform(img, sepia_filter)
        img_sepia = np.clip(img_sepia, 0, 255)

        # Apply a subtle vignette
        rows, cols = img.shape[:2]
        X_resultant_kernel = cv2.getGaussianKernel(cols, cols//2)
        Y_resultant_kernel = cv2.getGaussianKernel(rows, rows//2)
        resultant_kernel = Y_resultant_kernel * X_resultant_kernel.T
        mask = 255 * resultant_kernel / np.linalg.norm(resultant_kernel)
        img_vignette = np.copy(img_sepia)
        img_vignette[:,:,0] = img_vignette[:,:,0] * mask
        img_vignette[:,:,1] = img_vignette[:,:,1] * mask
        img_vignette[:,:,2] = img_vignette[:,:,2] * mask

        # Add some noise
        intensity = 0.05
        noise = np.random.normal(size=img_vignette.shape, scale=intensity*255)
        img_vintage = np.clip(img_vignette + noise, 0, 255).astype(np.uint8)

        # Save and display the vintage image
        vintage_filename = f"{input_text}_vintage.png"
        vintage_img = Image.fromarray(img_vintage)
        vintage_img.thumbnail((256, 256))  # Resize the image to fit the GUI
        vintage_img.save(vintage_filename)
        img_vintage_tk = ImageTk.PhotoImage(vintage_img)
        fx_image_label.config(image=img_vintage_tk)
        fx_image_label.image = img_vintage_tk

    except Exception as e:
        print("An error occurred during vintage filter:", e)

#----------------------------------------------------------------------------------------------------------------------

#thermal vision

def thermal_vision():
    try:
        # Read the image
        input_text = text_entry.get()
        image_filename = f"{input_text}.png"
        img = cv2.imread(image_filename)

        # Convert color space to Lab
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

        # Split the Lab image into L, a and b channels
        l, a, b = cv2.split(img_lab)

        # Apply a colormap and normalize the image
        img_thermal = cv2.applyColorMap(l, cv2.COLORMAP_JET)
        img_thermal = cv2.normalize(img_thermal, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # Save and display the thermal vision image
        thermal_vision_filename = f"{input_text}_thermal_vision.png"
        cv2.imwrite(thermal_vision_filename, img_thermal*255)
        img_thermal = Image.open(thermal_vision_filename)
        img_thermal.thumbnail((256, 256))
        img_thermal_tk = ImageTk.PhotoImage(img_thermal)
        fx_image_label.config(image=img_thermal_tk)
        fx_image_label.image = img_thermal_tk

    except Exception as e:
        print("An error occurred during thermal vision effect:", e)

#----------------------------------------------------------------------------------------------------------------------

#brighten

def brighten_image():
    try:
        # Read the image
        input_text = text_entry.get()
        image_filename = f"{input_text}.png"
        img = cv2.imread(image_filename)

        # Brighten the image - increase the scale factor to 1.5 for more brightness
        img_bright = cv2.convertScaleAbs(img, alpha=1.5, beta=0)

        # Save and display the brightened image
        bright_filename = f"{input_text}_bright.png"
        cv2.imwrite(bright_filename, img_bright)
        img_bright = Image.open(bright_filename)
        img_bright.thumbnail((256, 256))
        img_bright_tk = ImageTk.PhotoImage(img_bright)
        fx_image_label.config(image=img_bright_tk)
        fx_image_label.image = img_bright_tk

    except Exception as e:
        print("An error occurred during brightening:", e)

#----------------------------------------------------------------------------------------------------------------------

#darken

def darken_image():
    try:
        # Read the image
        input_text = text_entry.get()
        image_filename = f"{input_text}.png"
        img = cv2.imread(image_filename)

        # Darken the image - decrease the scale factor to 0.5 for more darkness
        img_dark = cv2.convertScaleAbs(img, alpha=0.5, beta=0)

        # Save and display the darkened image
        dark_filename = f"{input_text}_dark.png"
        cv2.imwrite(dark_filename, img_dark)
        img_dark = Image.open(dark_filename)
        img_dark.thumbnail((256, 256))
        img_dark_tk = ImageTk.PhotoImage(img_dark)
        fx_image_label.config(image=img_dark_tk)
        fx_image_label.image = img_dark_tk

    except Exception as e:
        print("An error occurred during darkening:", e)

#----------------------------------------------------------------------------------------------------------------------

# Rotate left

def rotate_left():
    try:
        input_text = text_entry.get()
        image_filename = f"{input_text}.png"
        img = cv2.imread(image_filename)
        
        # Perform rotation
        img_rotated_left = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Save and display the rotated image
        rotated_left_filename = f"{input_text}_rotated_left.png"
        cv2.imwrite(rotated_left_filename, img_rotated_left)
        img_rotated_left = Image.open(rotated_left_filename)
        img_rotated_left.thumbnail((256, 256))  
        img_rotated_left_tk = ImageTk.PhotoImage(img_rotated_left)
        fx_image_label.config(image=img_rotated_left_tk)
        fx_image_label.image = img_rotated_left_tk

    except Exception as e:
        print("An error occurred during rotating image to left:", e)

#----------------------------------------------------------------------------------------------------------------------

# Rotate right

def rotate_right():
    try:
        input_text = text_entry.get()
        image_filename = f"{input_text}.png"
        img = cv2.imread(image_filename)
        
        # Perform rotation
        img_rotated_right = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        # Save and display the rotated image
        rotated_right_filename = f"{input_text}_rotated_right.png"
        cv2.imwrite(rotated_right_filename, img_rotated_right)
        img_rotated_right = Image.open(rotated_right_filename)
        img_rotated_right.thumbnail((256, 256))  
        img_rotated_right_tk = ImageTk.PhotoImage(img_rotated_right)
        fx_image_label.config(image=img_rotated_right_tk)
        fx_image_label.image = img_rotated_right_tk

    except Exception as e:
        print("An error occurred during rotating image to right:", e)

#----------------------------------------------------------------------------------------------------------------------

# Dreamy Effect

def dreamy_effect():
    try:
        input_text = text_entry.get()
        image_filename = f"{input_text}.png"
        img = cv2.imread(image_filename)

        # Apply dreamy effect processing
        # Example code:
        # Apply Gaussian blur
        img_blurred = cv2.GaussianBlur(img, (25, 25), 0)

        # Increase image contrast
        img_contrast = cv2.convertScaleAbs(img_blurred, alpha=1.5, beta=0)

        # Apply color enhancement and saturation
        img_enhanced = cv2.cvtColor(img_contrast, cv2.COLOR_BGR2HSV)
        img_enhanced[..., 1] = img_enhanced[..., 1] * 1.2  # Increase saturation
        img_enhanced = cv2.cvtColor(img_enhanced, cv2.COLOR_HSV2BGR)

        # Save and display the dreamy effect image
        dreamy_filename = f"{input_text}_dreamy.png"
        cv2.imwrite(dreamy_filename, img_enhanced)
        img_dreamy = Image.open(dreamy_filename)
        img_dreamy.thumbnail((256, 256))
        img_dreamy_tk = ImageTk.PhotoImage(img_dreamy)
        fx_image_label.config(image=img_dreamy_tk)
        fx_image_label.image = img_dreamy_tk

    except Exception as e:
        print("An error occurred during dreamy effect:", e)
        
#----------------------------------------------------------------------------------------------------------------------

# grainy

def glitch_art():
    try:
        input_text = text_entry.get()
        image_filename = f"{input_text}.png"
        img = cv2.imread(image_filename)
        
        # Perform glitch art processing (e.g., data corruption, pixel manipulation, etc.)
        # Example code:
        # Apply data corruption effect
        img_corrupted = img.copy()
        rows, cols, _ = img.shape
        for i in range(rows):
            for j in range(cols):
                if random.random() < 0.05:  # Randomly corrupt pixels
                    img_corrupted[i, j] = [random.randint(0, 255) for _ in range(3)]
        
        # Save and display the glitch art image
        glitch_art_filename = f"{input_text}_grainy.png"
        cv2.imwrite(glitch_art_filename, img_corrupted)
        img_glitch_art = Image.open(glitch_art_filename)
        img_glitch_art.thumbnail((256, 256))
        img_glitch_art_tk = ImageTk.PhotoImage(img_glitch_art)
        fx_image_label.config(image=img_glitch_art_tk)
        fx_image_label.image = img_glitch_art_tk

    except Exception as e:
        print("An error occurred during glitch art effect:", e)

#----------------------------------------------------------------------------------------------------------------------

# abstract

def pop_culture_filter():
    try:
        input_text = text_entry.get()
        image_filename = f"{input_text}.png"
        img = cv2.imread(image_filename)
        
        # Apply pop culture filter processing (e.g., stylize image, add special effects, etc.)
        # Example code:
        # Apply cartoon filter
        cartoon_img = cv2.stylization(img, sigma_s=200, sigma_r=0.3)
        
        # Save and display the pop culture filtered image
        pop_culture_filename = f"{input_text}_pop_culture.png"
        cv2.imwrite(pop_culture_filename, cartoon_img)
        img_pop_culture = Image.open(pop_culture_filename)
        img_pop_culture.thumbnail((256, 256))
        img_pop_culture_tk = ImageTk.PhotoImage(img_pop_culture)
        fx_image_label.config(image=img_pop_culture_tk)
        fx_image_label.image = img_pop_culture_tk

    except Exception as e:
        print("An error occurred during pop culture filter effect:", e)

#----------------------------------------------------------------------------------------------------------------------

# A S C I I

def convert_image_to_ascii():
    try:
        input_text = text_entry.get()
        image_filename = f"{input_text}.png"

        img = Image.open(image_filename).convert('L')

        ascii_chars = " .':-;=+*#%@$"  # ASCII characters used to represent different intensity levels
        ascii_image = ""
        ascii_img_list = []

        width, height = img.size
        aspect_ratio = height / width
        new_width = 100
        new_height = int(aspect_ratio * new_width * 0.55)

        img = img.resize((new_width, new_height))

        division_factor = 256 / len(ascii_chars)

        for y in range(new_height):
            for x in range(new_width):
                pixel_value = img.getpixel((x, y))
                ascii_index = int(pixel_value / division_factor)
                ascii_image += ascii_chars[ascii_index]
                ascii_img_list.append(ascii_chars[ascii_index])
            ascii_image += "\n"
            ascii_img_list.append('\n')

        ascii_filename = f"{input_text}_ascii.txt"
        with open(ascii_filename, "w") as f:
            f.write(ascii_image)
        print(f"ASCII art saved as {ascii_filename}")

        # Create image from ASCII art
        font = ImageFont.load_default()
        ascii_img = Image.new('L', (font.getsize(' ')[0]*new_width, font.getsize(' ')[1]*new_height), color=255)
        d = ImageDraw.Draw(ascii_img)
        y_pos = 0
        for row in ''.join(ascii_img_list).split('\n'):
            d.text((0, y_pos), row, font=font, fill=0)
            y_pos += font.getsize(' ')[1]

        ascii_img_filename = f"{input_text}_ascii_img.png"
        ascii_img.save(ascii_img_filename)
        print(f"ASCII image saved as {ascii_img_filename}")

    except Exception as e:
        print("An error occurred during ASCII art conversion:", e)

#----------------------------------------------------------------------------------------------------------------------


# ASCII IMPORT function

def image_to_ascii(image_path):
    try:
        img = Image.open(image_path).convert('L')
        ascii_chars = " .:-=+*#%@"
        ascii_image = ""
        width, height = img.size
        aspect_ratio = height / width
        new_width = 100
        new_height = int(aspect_ratio * new_width * 0.55)
        img = img.resize((new_width, new_height))
        division_factor = 256 / len(ascii_chars)

        ascii_img_list = []

        for y in range(new_height):
            for x in range(new_width):
                pixel_value = img.getpixel((x, y))
                ascii_index = int(pixel_value / division_factor)
                ascii_image += ascii_chars[ascii_index]
                ascii_img_list.append(ascii_chars[ascii_index])
            ascii_image += "\n"
            ascii_img_list.append('\n')

        ascii_filename = image_path.split('.')[0] + "_ascii.txt"
        with open(ascii_filename, "w") as f:
            f.write(ascii_image)
        print(f"ASCII art saved as {ascii_filename}")

        # Create image from ASCII art
        font = ImageFont.load_default()
        ascii_img = Image.new('L', (font.getsize(' ')[0]*new_width, font.getsize(' ')[1]*new_height), color=255)
        d = ImageDraw.Draw(ascii_img)
        y_pos = 0
        for row in ''.join(ascii_img_list).split('\n'):
            d.text((0, y_pos), row, font=font, fill=0)
            y_pos += font.getsize(' ')[1]

        ascii_img_filename = image_path.split('.')[0] + "_ascii_img.png"
        ascii_img.save(ascii_img_filename)
        print(f"ASCII image saved as {ascii_img_filename}")

    except Exception as e:
        print("An error occurred during ASCII art conversion:", e)

# Function to open file dialog and get the image path
def import_image():
    filename = filedialog.askopenfilename()
    return filename

def import_img_to_ascii():
    image_path = import_image()
    image_to_ascii(image_path)


#----------------------------------------------------------------------------------------------------------------------

# A U T O   C R O P

def auto_crop():
    try:
        input_text = text_entry.get()
        image_filename = f"{input_text}.png"
        img = Image.open(image_filename)

        # Sizes to resize and crop to with their corresponding names
        sizes_and_names = [((975, 180), f"{input_text}_bandcamp"), 
                   ((1500, 500), f"{input_text}_twitter"), 
                   ((1640, 924), f"{input_text}_facebook")]

        for size, name in sizes_and_names:
        # We resize the image with the same aspect ratio
            img_resized = img.resize(size, Image.ANTIALIAS)  # size is now a tuple
        # Then we crop it to the exact size
            cropped = img_resized.crop((0, 0, size[0], size[1]))
        # Save the cropped image with the corresponding name
            cropped.save(f"{name}.png")
        # Resize the cropped image to fit the GUI
            cropped.thumbnail((256, 256))
            cropped_tk = ImageTk.PhotoImage(cropped)
            fx_image_label.config(image=cropped_tk)
            fx_image_label.image = cropped_tk

    except Exception as e:
        print("An error occurred during auto crop:", e)


#======================================================================================================================
#======================================================================================================================


# A D D    F O N T    T O    I M A G E

def open_image():
    global image_on_canvas
    global canvas

    image_filename = f"{text_entry.get()}.png"
    img = Image.open(image_filename)
    img.thumbnail((512, 512))  # Resize the image to fit the GUI

    image_on_canvas = ImageTk.PhotoImage(img)
    canvas = tk.Canvas(root, width=img.width, height=img.height)
    canvas.create_image(0, 0, anchor='nw', image=image_on_canvas)
    canvas.grid(row=0, column=0)


def add_text_to_image(img, text, position, font_path, font_size, color):
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    return img


def add_text_on_image():
    try:
        image_filename = f"{text_entry.get()}.png"
        img = Image.open(image_filename)

        text = text_on_image_entry.get()
        font = os.path.expanduser(f"~/Library/Fonts/{font_var.get()}.ttf")
        font_size = int(font_size_var.get())
        color = color_var.get()

        position = (int(x_position_var.get()), int(y_position_var.get()))

        updated_image = add_text_to_image(img, text, position, font, font_size, color)

        # Save and display the updated image
        updated_image_filename = f"{text_entry.get()}_with_font.png"
        updated_image.save(updated_image_filename)
        updated_image.thumbnail((256, 256))  # Resize the image to fit the GUI
        updated_image_tk = ImageTk.PhotoImage(updated_image)
        fx_image_label.config(image=updated_image_tk)
        fx_image_label.image = updated_image_tk

    except Exception as e:
        print("An error occurred during adding text to the image:", e)


#======================================================================================================================
#======================================================================================================================

# U P S C A L E    S E T T I N G S


# U P S C A L E    S E T T I N G S

# U P S C A L E    S E T T I N G S

def upscale_image():
    global current_img
    input_text = text_entry.get()  # Retrieve user input from the text entry widget
    image_filename = f"{input_text}.png"

    # Check if the image file exists before trying to open it
    if current_img:
        img = current_img
        # Continue with the rest of the code
    else:
        print("Image not loaded.")  # Print a message if no image is loaded

    # Call the upscaling process using the Stability AI API
    answers = stability_api.upscale(init_image=img)

    # Process the response and save the upscaled image
    for resp in answers:
        for artifact in resp.artifacts:
            if artifact.finish_reason == generation.FILTER:
                warnings.warn("Your request activated the API's safety filters and could not be processed."
                              "Please submit a different image and try again.")
            if artifact.type == generation.ARTIFACT_IMAGE:
                big_img = Image.open(io.BytesIO(artifact.binary))

                # Get current timestamp and convert it to a string
                # The timestamp format is YearMonthDay_HourMinuteSecond
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

                # Save the upscaled image to a local file with a timestamp
                big_img.save(f"{timestamp}_{input_text}_upscaled.png")

                


#======================================================================================================================
#======================================================================================================================


# A N I M A T I O N   S E T T I N G S

def update_image(cap, label):
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (256, 256))  # Resize the frame to desired dimensions
        img = Image.fromarray(frame)
        img_tk = ImageTk.PhotoImage(img)
        label.config(image=img_tk)
        label.image = img_tk
        label.after(1, update_image, cap, label)
    else:
        cap.release()

#----------------------------------------------------------------------------------------------------------------------

# stitch vid

def stitch_frames_to_video(frames_directory, output_filename, fps):
    frames = []
    frame_files = os.listdir(frames_directory)

    def try_parse_int(s, base=10, val=None):
        try:
            return int(s, base)
        except ValueError:
            return val

    frame_files.sort(key=lambda x: try_parse_int(x.split('_')[1].split('.')[0], val=float('inf')))

    for filename in frame_files:
        if filename.endswith('.png') or filename.endswith('.jpg'):  # only process expected image file types
            frames.append(cv2.imread(os.path.join(frames_directory, filename)))
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
    for frame in frames:
        video_writer.write(frame)
    video_writer.release()

#----------------------------------------------------------------------------------------------------------------------

def generate_video():
    start_animation_text = animation_start_entry.get()
    mid_animation_text = animation_mid_entry.get()
    mid_animation_text2 = animation_mid_entry2.get()
    mid_animation_text3 = animation_mid_entry3.get()
    end_animation_text = animation_end_entry.get()
    animation_length = int(animation_length_entry.get())

    start_frame = int(start_frame_entry.get())
    mid_frame = int(mid_frame_entry.get())
    mid_frame2 = int(mid_frame_entry2.get())
    mid_frame3 = int(mid_frame_entry3.get())
    end_frame = int(end_frame_entry.get())

    # Get seed from GUI
    seed = int(seed_entry.get())

    args = AnimationArgs()
    args.interpolate_prompts = True
    args.locked_seed = True
    args.max_frames = animation_length
    args.seed = seed  # use the seed from GUI
    args.strength_curve = "0:(0)"
    args.diffusion_cadence_curve = "0:(4)"
    args.cadence_interp = "film"

    # Get the selected preset from the combobox
    selected_preset = preset_combobox.get()

    # Set the preset parameter to the selected value
    args.preset = selected_preset

    args.fps = int(fps_entry.get())

    # Check the state of the checkboxes and set translation parameters accordingly
    args.translation_z = "0:(1)" if zoom_state.get() else "0:(0)"
    args.translation_y = "0:(1)" if up_state.get() else "0:(-0.5)" if down_state.get() else "0:(0)"
    args.translation_x = "0:(1)" if right_state.get() else "0:(-0.5)" if left_state.get() else "0:(0)"

    animation_prompts = {
        start_frame: start_animation_text,
        mid_frame: mid_animation_text,
        mid_frame2: mid_animation_text2,
        mid_frame3: mid_animation_text3,
        end_frame: end_animation_text,
    }

    negative_prompt = "nudity, naked, violence, blood, horror, watermark, logo, sex, guns"

    # Create a unique name based on the start prompt, selected preset, and current time
    unique_id = int(time.time())  # Get the current time in seconds since the epoch as an integer
    folder_name = f"{start_animation_text}_{selected_preset}_{unique_id}"
    video_name = f"{folder_name}.mp4"

    animator = Animator(
        api_context=api_context,
        animation_prompts=animation_prompts,
        negative_prompt=negative_prompt,
        args=args,
        out_dir=folder_name  # Use the unique folder name
    )

    try:
        for _ in tqdm(animator.render(), total=args.max_frames):
            pass
    except Exception as e:
        print(f"Animation terminated early due to exception: {e}")

    output_video_filename = video_name
    stitch_frames_to_video(folder_name, output_video_filename, animation_length)  # Use the unique video name

    # Open the video file
    cap = cv2.VideoCapture(output_video_filename)

    # Start the update process
    update_image(cap, video_label)

    return output_video_filename

play_state = True  # a global flag to control video playback

#----------------------------------------------------------------------------------------------------------------------

def update_image(cap, label):
    global play_state
    if play_state:  # only update if play_state is True
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (256, 256))  # Resize the frame to desired dimensions
            img = Image.fromarray(frame)
            img_tk = ImageTk.PhotoImage(img)
            label.config(image=img_tk)
            label.image = img_tk
            label.after(1, update_image, cap, label)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to the start of the video if it ends


#======================================================================================================================
#======================================================================================================================

# video chaos

def video_chaos(word_library):
    random_sets = []
    for _ in range(5):
        num_words = random.randint(4, 7)
        random_set = random.sample(word_library, num_words)
        random_sets.append(", ".join(random_set))
    return random_sets

def video_chaos_button_click():
    min_words = 4
    max_words = 8
    num_text_fields = 5
    random_frames = sorted(random.sample(range(1, 25), 4))  # Generate 4 random frames in chronological order
    random_words_sets = video_chaos(sample_words)  # Use sample_words instead of word_library
    random_seed = random.randint(5, 15)  # Generate a random seed count between 5 and 15

    # Populate the text fields with random words
    animation_start_entry.delete(0, tk.END)
    animation_start_entry.insert(0, random_words_sets[0])
    start_frame_entry.delete(0, tk.END)
    start_frame_entry.insert(0, 0)  # Start frame is always 0

    animation_mid_entry.delete(0, tk.END)
    animation_mid_entry.insert(0, random_words_sets[1])
    mid_frame_entry.delete(0, tk.END)
    mid_frame_entry.insert(0, random_frames[0])

    animation_mid_entry2.delete(0, tk.END)
    animation_mid_entry2.insert(0, random_words_sets[2])
    mid_frame_entry2.delete(0, tk.END)
    mid_frame_entry2.insert(0, random_frames[1])

    animation_mid_entry3.delete(0, tk.END)
    animation_mid_entry3.insert(0, random_words_sets[3])
    mid_frame_entry3.delete(0, tk.END)
    mid_frame_entry3.insert(0, random_frames[2])

    animation_end_entry.delete(0, tk.END)
    animation_end_entry.insert(0, random_words_sets[4])
    end_frame_entry.delete(0, tk.END)
    end_frame_entry.insert(0, random_frames[3])

    # Add the entry field for frame count
    frame_count_label_text = tk.Label(animation_frame_text, text="Total Frames:")
    frame_count_entry = tk.Entry(animation_frame_text, width=3)

    # Calculate the total frame count based on the "End Frame" entry
    total_frames = int(end_frame_entry.get())

    # Auto-set frame count to the total_frames
    animation_length_entry.delete(0, tk.END)
    animation_length_entry.insert(0, total_frames)

    # Calculate the total frame count based on the "End Frame" entry
    total_frames = int(end_frame_entry.get())

    # Auto-set seed count to a random number between 5 and 15
    seed_entry.delete(0, tk.END)
    seed_entry.insert(0, random_seed)

    # Auto-set FPS to 12
    fps_entry.delete(0, tk.END)
    fps_entry.insert(0, 12)

    # Auto-set frame count to the total_frames
    frame_count_entry.delete(0, tk.END)
    frame_count_entry.insert(0, total_frames)

#======================================================================================================================
#======================================================================================================================


# I M A G E   T O   V I D E O

# Declare args globally
args = AnimationArgs()


def import_video():
    filename = filedialog.askopenfilename()  # Get the file path
    # Load the file into your program
    args.init_image = filename  # Assign the filename directly to args

    # Load the image
    load = Image.open(filename)
    # Resize the image using thumbnail or another method if needed
    load.thumbnail((100, 100))  # example for a 100x100 thumbnail
    render = ImageTk.PhotoImage(load)
    
    # Put the image on a label
    img = tk.Label(animation_frame_image, image=render)
    img.image = render
    
    # Position the label
    img.grid(row=0, column=1)

# Now the generate_image_to_video function will use multiple images at different frames
def generate_image_to_video():
    # Set up animation args
    args.interpolate_prompts = True
    args.locked_seed = True
    args.max_frames = 48
    args.seed = 42
    args.strength_curve = "0:(0)"
    args.diffusion_cadence_curve = "0:(4)"
    args.cadence_interp = "film"

    # Set up animation image prompts
    animation_images = {
        int(start_frame_entry_image.get()): start_image_entry.get(),
        int(mid_frame_entry_image.get()): mid_image_entry.get(),
        int(end_frame_entry_image.get()): end_image_entry.get(),
    }

    negative_prompt = ""  # Add any negative prompts here

    animator = Animator(
        api_context=api_context,
        animation_prompts=animation_images,  # using animation_images instead of animation_prompts
        negative_prompt=negative_prompt,
        args=args,
        out_dir="output"
    )

    try:
        for _ in tqdm(animator.render(), total=args.max_frames):
            pass
    except Exception as e:
        print(f"Animation terminated early due to exception: {e}")

    # Get the name of the input image file
    input_image_name = os.path.splitext(os.path.basename(args.init_image))[0]
    # Create output filename by appending " - animated" to the input image name
    output_filename = f"{input_image_name} - animated.mp4"
    
    create_video_from_frames(animator.out_dir, output_filename, fps=24)

def create_video_from_frames(frames_directory, output_filename, fps):
    # Get the frame files from the directory
    frame_files = sorted([os.path.join(frames_directory, f) for f in os.listdir(frames_directory) if os.path.isfile(os.path.join(frames_directory, f))])

    # Read the frames
    frames = [cv2.imread(f) for f in frame_files if f.endswith(".png") or f.endswith(".jpg")]

    # Ensure there are frames in the directory
    if len(frames) > 0:
        # Get the shape of the frames
        height, width, _ = frames[0].shape

        # Define the codec using VideoWriter_fourcc and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

        # Write the frames to the file
        for i in range(len(frames)):
            out.write(frames[i])

        # Release the VideoWriter
        out.release()
    else:
        print(f"No frames in directory {frames_directory} to create video from.")
        

#======================================================================================================================
#======================================================================================================================


# G P T   C H A O S S 

def random_chaos():
    # Generate 5 sentences with ChatGPT
    sentences = []
    for i in range(5):
        messages = [
            {"role": "system", "content": "You will act as if you are creative screenwriter."},
            {"role": "user", "content": "Invent movie scene in 1 sentence that 15 words or less:"}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            max_tokens=100,
            temperature=1
        )
        sentences.append(response['choices'][0]['message']['content'].strip())

    # Insert the sentences into the text fields
    text_fields = [animation_start_entry, animation_mid_entry, animation_mid_entry2, animation_mid_entry3, animation_end_entry]
    for i in range(5):
        text_fields[i].delete(0, tk.END)
        text_fields[i].insert(0, sentences[i])


#======================================================================================================================
#======================================================================================================================


 #  V I D E O    P L A Y B A C K

def update_canvas():
    global playback_status, video, filepath, photo
    if playback_status == 'play':
        ret, frame = video.read()
        if ret:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            photo = ImageTk.PhotoImage(img)
            canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            video_player_frame.after(15, update_canvas)  # Only reschedule if the video is playing and a frame was successfully read
        elif not ret:
            video = cv2.VideoCapture(filepath)  # Reload the video if it finished playing
            update_canvas()  # Immediately try to update the canvas

def play_video():
    global playback_status
    playback_status = 'play'
    update_canvas()

def pause_video():
    global playback_status
    playback_status = 'pause'
    # Do not call update_canvas here

def stop_video():
    global playback_status
    playback_status = 'stop'
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rewind video
    update_canvas()  # Immediately update the canvas

def upload_video():
    global video, canvas, filepath
    filepath = filedialog.askopenfilename()
    if filepath:
        video = cv2.VideoCapture(filepath)
        # Update the size of the canvas to fit the video
        canvas.config(width=new_width, height=new_height)


#======================================================================================================================
#======================================================================================================================


# M I D I   G E N E R A T I O N   P A R A M E T E R S

def generate_midi_from_image():
    input_text = text_entry.get()  # Retrieve user input from the text entry widget
    image_filename = f"{input_text}.png"

    if os.path.exists(image_filename):
        midi_filename = f"{input_text}_midi1.mid"
        image = Image.open(image_filename)
        calculate_parameters_and_create_midi(image, midi_filename)
        print(f"MIDI file saved as {midi_filename}")
    else:
        print("Image file not found.")


def image_to_midi(image_path, midi_path):
    # Load the image
    image = Image.open(image_path).convert('L')

#----------------------------------------------------------------------------------------------------------------------

# MIDI ALGORITHM 1

def calculate_parameters_and_create_midi(image, midi_path):
    musical_parameters = {}

    # Create a new MIDI file with 3 tracks
    midi = MIDIFile(3)

    # Set the instrument for each track (using General MIDI program numbers)
    midi.addProgramChange(0, 0, 0, 89)  # Pad 2 (warm) for the Red channel
    midi.addProgramChange(1, 1, 0, 80)  # Lead 1 (square) for the Green channel
    midi.addProgramChange(2, 2, 0, 38)  # Synth Bass 1 for the Blue channel

    # Time counter
    time = 0

    # Define a scale for each instrument (C Major scale in three octaves)
    major_scale = [60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79, 81, 82, 84, 85, 87, 89, 91]
    minor_scale = [60, 62, 63, 65, 67, 68, 70, 72, 75, 76, 77, 79, 82, 84, 85, 87, 88, 90, 91]

    # Define a separate parameter for note duration
    base_duration = 0.25  # Adjust this value to control the base duration of notes

    # Calculate the average brightness of the image
    average_color = [sum(c) / len(c) for c in zip(*image.getdata())]
    average_brightness = sum(average_color) / 3

    # Determine whether the composition should be major or minor
    is_major = average_brightness >= 128  # Adjust the threshold as desired

    # Select the appropriate scale based on major or minor
    scale = major_scale if is_major else minor_scale

    # For each pixel in the image
    pixel_count = 0  # Counter to keep track of the processed pixels

    # Get the width and height of the image
    width, height = image.size

    # Pixel limit for MIDI conversion
    total_pixels = 42000

    for y in range(height):
        for x in range(width):
            # Increment the pixel counter
            pixel_count += 1

            # Check if the maximum number of pixels to process is reached
            if pixel_count > total_pixels:
                break

            # Get the RGB values of the pixel
            r, g, b = image.getpixel((x, y))

            # Map each RGB value to an index in the scale
            r_index = int(r / 256 * len(scale))
            g_index = int(g / 256 * len(scale))
            b_index = int(b / 256 * len(scale))

            # Get the MIDI note numbers from the scale
            r_note = scale[r_index]
            g_note = scale[g_index]
            b_note = scale[b_index]

            # Calculate the note duration based on a separate factor or parameter
            duration = max(base_duration * (r + g + b) / (3 * 256), 0.1)  # Ensure duration is at least 0.1

            # Add a note to the MIDI file for each instrument
            midi.addNote(0, 0, r_note, time, duration=duration, volume=100)  # Pad synth track
            midi.addNote(1, 1, g_note, time, duration=duration, volume=100)  # Lead synth track
            midi.addNote(2, 2, b_note, time, duration=duration, volume=100)  # Bass synth track

            # Increment the time counter
            time += duration

        if pixel_count > total_pixels:
            break

    # Calculate the average color of the image and set the tempo
    average_color = [sum(c) / len(c) for c in zip(*image.getdata())]
    average_brightness = sum(average_color) / 3
    tempo = 60 + (average_brightness / 255) * 120  # Tempo will range from 60 to 180 BPM
    midi.addTempo(0, 0, tempo)
    midi.addTempo(1, 0, tempo)
    midi.addTempo(2, 0, tempo)

    # Write the MIDI file
    midi_filename = midi_path + '_midi2.mid'
    with open(midi_filename, 'wb') as f:
        midi.writeFile(f)
#----------------------------------------------------------------------------------------------------------------------

# MIDI ALGORITHM 2

def calculate_parameters_and_create_advanced_midi(image, midi_path):
    # Create a new MIDI file with 3 tracks
    midi = MIDIFile(3)

    # Set the instrument for each track (using General MIDI program numbers)
    midi.addProgramChange(0, 0, 0, 87)  # Lead 8 (bass + lead) for the Red channel
    midi.addProgramChange(1, 1, 0, 54)  # Voice Synth for the Green channel
    midi.addProgramChange(2, 2, 0, 62)  # Brass Synth for the Blue channel

    # Time counter
    time = 0

    # Define a scale for each instrument (C Major scale in three octaves)
    major_scale = [i - 24 for i in [60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79, 81, 82, 84, 85, 87, 89, 91]]
    minor_scale = [i - 24 for i in [60, 62, 63, 65, 67, 68, 70, 72, 74, 76, 77, 79, 81, 82, 84, 85, 87, 89, 91]]

    # Define a separate parameter for note duration
    base_duration = 0.5

    # Calculate the average brightness of the image
    average_color = [sum(c) / len(c) for c in zip(*image.getdata())]
    average_brightness = sum(average_color) / 3

    # Determine whether the composition should be major or minor
    is_major = average_brightness >= 128

    # Select the appropriate scale based on major or minor
    scale = major_scale if is_major else minor_scale

    # For each pixel in the image
    pixel_count = 0

    # Get the width and height of the image
    width, height = image.size

    # Pixel limit for MIDI conversion
    total_pixels = 42000

    # Calculate the saturation of the image
    im_stat = ImageStat.Stat(image)
    saturation = np.std(im_stat.mean)

    # Calculate volume based on saturation (range between 50 and 127)
    volume = int(50 + (saturation / np.sqrt((255**2)/3)) * 77)

    # Calculate complexity and colorfulness of the image
    complexity = np.std(image.histogram())
    colorfulness = np.mean([abs(pixel - average_color) for pixel in image.getdata()])

    # Map complexity to velocity (range 50-127) and colorfulness to pan (range -1 to 1)
    velocity = int(50 + complexity * 77 / 255)
    pan = colorfulness * 2 / 255 - 1

    for y in range(height):
        for x in range(width):
            # Increment the pixel counter
            pixel_count += 1

            # Check if the maximum number of pixels to process is reached
            if pixel_count > total_pixels:
                break

            # Get the RGB values of the pixel
            r, g, b = image.getpixel((x, y))

            # Map each RGB value to an index in the scale
            r_index = int(r / 256 * len(scale))
            g_index = int(g / 256 * len(scale))
            b_index = int(b / 256 * len(scale))

            # Get the MIDI note numbers from the scale
            r_note = scale[r_index]
            g_note = scale[g_index]
            b_note = scale[b_index]

            # Calculate the note duration based on a separate factor or parameter
            duration = max(base_duration * (r + g + b) / (3 * 256), 0.1)

            # Add a note to the MIDI file for each instrument with velocity and pan
            midi.addNote(0, 0, r_note, time, duration=duration, volume=volume, velocity=velocity, pan=pan)
            midi.addNote(1, 1, g_note, time, duration=duration, volume=volume, velocity=velocity, pan=pan)
            midi.addNote(2, 2, b_note, time, duration=duration, volume=volume, velocity=velocity, pan=pan)

            # Increment the time counter
            time += duration

        if pixel_count > total_pixels:
            break

    # Adjust tempo based on image brightness but slower than original
    tempo = 30 + (average_brightness / 255) * 90
    midi.addTempo(0, 0, tempo)
    midi.addTempo(1, 0, tempo)
    midi.addTempo(2, 0, tempo)

    # Write the MIDI file
    midi_filename = midi_path + '_midi2.mid'
    with open(midi_filename, 'wb') as f:
        midi.writeFile(f)


#======================================================================================================================
#======================================================================================================================


# A U D I O   G E N E R A T I O N   S E T T I N G S

def generate_audio_from_image():
    input_text = text_entry.get()  # Retrieve user input from the text entry widget
    image_filename = f"{input_text}.png"

    if os.path.exists(image_filename):
        wav_filename = f"{input_text}_audio.wav"
        image = Image.open(image_filename)
        generate_audio_from_image_impl(image, wav_filename)
        print(f"Audio file saved as {wav_filename}")
    else:
        print("Image file not found.")

#----------------------------------------------------------------------------------------------------------------------

#IMAGE TO AUDIO ALGORITHM 1

def generate_audio_from_image_impl(image, wav_filename):
    # Analyze the image parameters and map them to audio parameters
    pixel_data = np.array(image)
    brightness = np.mean(pixel_data) / 255.0
    contrast = np.std(pixel_data) / 255.0

    # Calculate the average color of the image and set the tempo
    average_color = [sum(c) / len(c) for c in zip(*image.getdata())]
    average_brightness = sum(average_color) / 3

    # Determine whether the composition should be major or minor
    is_major = average_brightness >= 128  # Adjust the threshold as desired

    # Select the appropriate scale based on major or minor
    scale = [72, 76, 79, 84, 88, 91, 96, 100, 103] if is_major else [72, 75, 79, 84, 87, 91, 96, 99, 103]  # One octave higher

    # Initialize an empty audio segment
    audio = AudioSegment.empty()

    # Time counter
    time = 0

    # Maximum audio duration
    max_duration = 10  # in seconds

    # Pixel limit for Audio conversion
    total_pixels = 1000

    # Get the width and height of the image
    width, height = image.size

    for y in range(height):
        for x in range(width):
            # Check if the maximum duration is reached
            if time >= max_duration:
                break

            # Get the RGB values of the pixel
            r, g, b = image.getpixel((x, y))

            # Map each RGB value to an index in the scale
            r_index = int(r / 256 * len(scale))
            g_index = int(g / 256 * len(scale))
            b_index = int(b / 256 * len(scale))

            # Get the MIDI note numbers from the scale
            r_note = scale[r_index]
            g_note = scale[g_index]
            b_note = scale[b_index]

            # Calculate the note duration based on a separate factor or parameter
            duration = max((r + g + b) / (3 * 256), 0.1)  # Ensure duration is at least 0.1

            # Generate audio segment for each pixel
            try:
                note = Sine(r_note).to_audio_segment(duration=int(duration * 1000))  # Adjust duration scaling factor as needed
                audio = audio.append(note, crossfade=0)
                time += duration  # Increment the time counter
            except Exception as e:
                print(f"Error in generating audio for pixel ({x}, {y}): {str(e)}")

        if time >= max_duration:
            break

    # Apply effects to the track
    audio = audio.fade_in(1000).fade_out(1000)  # Add fade-in and fade-out to audio track

    # Assume the audio file path is stored in `wav_filename`
    global audio_file_path  # Declare a global variable
    audio_file_path = wav_filename

    # Export the audio as a WAV file
    audio_filename = wav_filename + '_audio2.wav'
    try:
        audio.export(audio_filename, format="wav")
        print(f"Audio file saved as {audio_filename}")
    except Exception as e:
        print(f"Error in saving audio: {str(e)}")

#----------------------------------------------------------------------------------------------------------------------

#IMAGE TO AUDIO ALGORITHM 2

def generate_audio_from_image_impl2(image, wav_filename):
    # Analyze the image parameters and map them to audio parameters
    pixel_data = np.array(image)
    median_color_value = np.median(pixel_data) / 255.0
    contrast = np.std(pixel_data) / 255.0

    # Initialize an empty audio segment
    audio = AudioSegment.empty()

    # Time counter
    time = 0

    # Maximum audio duration
    max_duration = 15  # in seconds

    # Get the width and height of the image
    width, height = image.size

    for y in range(height):
        for x in range(width):
            # Check if the maximum duration is reached
            if time >= max_duration:
                break

            # Get the RGB values of the pixel
            r, g, b = image.getpixel((x, y))

            # Determine frequency based on the RGB values
            frequency = 440 + 10 * (r - g - b)  # A440 + offset

            # Calculate the note duration based on contrast
            duration = contrast * max((r + g + b) / (3 * 256), 0.1)  # Ensure duration is at least 0.1

            # Determine the type of waveform to use based on dominant color
            if r > g and r > b:
                waveform = Sine(frequency)
            elif g > r and g > b:
                waveform = Square(frequency)
            else:
                waveform = Sawtooth(frequency)

            # Generate audio segment for each pixel
            try:
                note = waveform.to_audio_segment(duration=int(duration * 1000))  # Adjust duration scaling factor as needed
                note = note + -20 * (1 - median_color_value)  # Adjust volume based on color median
                audio = audio.append(note, crossfade=0)
                time += duration  # Increment the time counter
            except Exception as e:
                print(f"Error in generating audio for pixel ({x}, {y}): {str(e)}")

        if time >= max_duration:
            break

    # Apply effects to the track
    audio = audio.fade_in(1000).fade_out(1000)  # Add fade-in and fade-out to audio track

    # Assume the audio file path is stored in `wav_filename`
    global audio_file_path  # Declare a global variable
    audio_file_path = wav_filename

    # Export the audio as a WAV file
    audio_filename = wav_filename + '_audio3.wav'
    try:
        audio.export(audio_filename, format="wav")
        print(f"Audio file saved as {audio_filename}")
    except Exception as e:
        print(f"Error in saving audio: {str(e)}")

#======================================================================================================================
#======================================================================================================================


# C H A T B O T    F U N C T I O N S 


def get_chat_response():
    current_tab_index = notebook.index(notebook.select())
    user_input = user_inputs[current_tab_index]
    chat_log = chat_logs[current_tab_index]
    conversation = conversations[current_tab_index]

    # Get user input
    message = user_input.get()

    # Clear the input field
    user_input.delete(0, tk.END)

    # Append user message to chat and conversation history
    chat_log.config(state=tk.NORMAL)
    chat_log.insert(tk.END, "You:   ", "user_name")
    chat_log.insert(tk.END, message + "\n", "user_answer")
    chat_log.config(state=tk.DISABLED)
    conversation.append({"role": "user", "content": message})

    # Get chatbot response using the Chat API
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=conversation
    )

    # Extract the chatbot response
    chatbot_response = response['choices'][0]['message']['content'].strip()

    # Append chatbot response to chat and conversation history
    chat_log.config(state=tk.NORMAL)
    #chat_log.insert(tk.END, "\n")
    chat_log.insert(tk.END, "GPT:   ", "bot_name")
    chat_log.insert(tk.END, chatbot_response + "\n", "bot_answer")
    chat_log.insert(tk.END, "\n\n")  # Double line break
    chat_log.config(state=tk.DISABLED)
    conversation.append({"role": "assistant", "content": chatbot_response})


    # Get chatbot response using the Chat API
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=conversation
    )

def save_specific_conversation(tab_index):
    # Ensure the directory exists
    if not os.path.exists('conversations'):
        os.makedirs('conversations')

    conversation = conversations[tab_index]
    # Use current datetime to create a unique filename
    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(f'conversations/conversation_{tab_index}_{timestamp_str}.json', 'w') as f:
        # Use json.dump to write the conversation to the file
        json.dump(conversation, f)

def load_specific_conversation(tab_index):
    # Open a file dialog for JSON files
    file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])

    # If a file was selected
    if file_path:
        with open(file_path, 'r') as f:
            # Load the conversation from the file
            loaded_conversation = json.load(f)

        # Set the current tab's conversation to the loaded conversation
        conversations[tab_index] = loaded_conversation

        # Update the chat log to match the loaded conversation
        chat_log = chat_logs[tab_index]
        chat_log.config(state=tk.NORMAL)
        chat_log.delete(1.0, tk.END)  # Clear the chat log
        for message in loaded_conversation:
            chat_log.insert(tk.END, f"{message['role'].capitalize()}: {message['content']}\n")
        chat_log.config(state=tk.DISABLED)

def clear_specific_conversation(tab_index):
    # Clear the chat log
    chat_log = chat_logs[tab_index]
    chat_log.config(state=tk.NORMAL)
    chat_log.delete(1.0, tk.END)
    chat_log.config(state=tk.DISABLED)
    
    # Clear the conversation history
    conversations[tab_index] = []
    

#======================================================================================================================
#======================================================================================================================


# N O T E P A D    F U N C T I O N S

class NotepadSketchpad(tk.Frame):
    def __init__(self, parent=None, **kwargs):
        tk.Frame.__init__(self, parent, **kwargs)

        # Create the notepad
        self.notepad = tk.Text(self, width=40, height=20)
        self.notepad.pack(side='left', fill='both', expand=True)

    def toggle_italic(self):
        current_tags = self.notepad.tag_names('sel.first')
        if 'italic' in current_tags:
            self.notepad.tag_remove('italic', 'sel.first', 'sel.last')
        else:
            self.notepad.tag_add('italic', 'sel.first', 'sel.last')
            self.notepad.tag_configure('italic', font=('TkDefaultFont', 11, 'italic'))

    def toggle_bold(self):
        current_tags = self.notepad.tag_names('sel.first')
        if 'bold' in current_tags:
            self.notepad.tag_remove('bold', 'sel.first', 'sel.last')
        else:
            self.notepad.tag_add('bold', 'sel.first', 'sel.last')
            self.notepad.tag_configure('bold', font=('TkDefaultFont', 11, 'bold'))

    def toggle_underline(self):
        current_tags = self.notepad.tag_names('sel.first')
        if 'underline' in current_tags:
            self.notepad.tag_remove('underline', 'sel.first', 'sel.last')
        else:
            self.notepad.tag_add('underline', 'sel.first', 'sel.last')
            self.notepad.tag_configure('underline', underline=True)
    def select_color(self):
        color = colorchooser.askcolor()[1]
        if color:
            self.notepad.tag_configure("colored", foreground=color)
            self.notepad.tag_add("colored", "1.0", "end")
            sketchpad.pen_color = color

    def toggle_color(self):
        current_tags = self.notepad.tag_names('sel.first')
        if 'colored' in current_tags:
            self.notepad.tag_remove('colored', 'sel.first', 'sel.last')
        else:
            self.select_color()
    def save_notepad_contents(self):
        contents = self.notepad.get("1.0", 'end-1c')  # Get the contents of the notepad
        current_time = datetime.datetime.now()  # Get the current date and time
        filename = current_time.strftime("%Y%m%d_%H%M%S_notepad.txt")  # Format date and time as a string, append '_notepad.txt'
        with open(filename, 'w') as file:  # Open a file in write mode with the generated filename
            file.write(contents)  # Write the contents to the file


#======================================================================================================================
#======================================================================================================================


# S K E T C H P A D   F U N C T I O N S

class Sketchpad(tk.Frame):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        self.canvas = tk.Canvas(self, bg='white')
        self.canvas.pack(fill='both', expand=True)

        self.pen_color = 'black'
        self.pen_size = 5  # Medium brush size

        self.drawing = False
        self.last_x = 0
        self.last_y = 0

        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

    def select_color(self):
        color = colorchooser.askcolor()[1]
        if color:
            self.pen_color = color
    def start_drawing(self, event):
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y
    def draw(self, event):
        if self.drawing:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, fill=self.pen_color,
                                    width=self.pen_size)
            self.last_x = event.x
            self.last_y = event.y
    def stop_drawing(self, event):
        self.drawing = False
    def save_sketch_as_jpg(self):
        try:
            x = self.canvas.winfo_rootx() + self.canvas.winfo_x()
            y = self.canvas.winfo_rooty() + self.canvas.winfo_y()
            x1 = x + self.canvas.winfo_width()
            y1 = y + self.canvas.winfo_height()

            current_time = datetime.datetime.now()  # Get the current date and time
            filename = current_time.strftime("%Y%m%d_%H%M%S_sketch.jpg")  # Format date and time as a string, append '_sketch.jpg'

            # Resize the image before saving
            ImageGrab.grab(bbox=(x, y, x1, y1)).convert("RGB").resize((512, 512)).save(filename, 'JPEG')

        except Exception as e:
            messagebox.showerror('Save Sketch', f'Error occurred while saving the sketch:\n{str(e)}')
    def clear_sketch(self):
        self.canvas.delete("all")

    def set_brush_size(self, size):
        self.pen_size = size

        
#======================================================================================================================
#======================================================================================================================


# G P T   N O T E B O O K    R A N D O M I Z E R S

def generate_idea_scenes():
    text = generate_ideas_text("movie_scene")
    notepad_sketchpad.notepad.insert(tk.END, text + "\n")

def generate_idea_styles():
    text = generate_ideas_text("art_and_photo_styles")
    notepad_sketchpad.notepad.insert(tk.END, text + "\n")

def generate_idea_adjectives():
    text = generate_ideas_text("uncommon_adj")
    notepad_sketchpad.notepad.insert(tk.END, text + "\n")

def generate_idea_trinkets():
    text = generate_ideas_text("trinkets")
    notepad_sketchpad.notepad.insert(tk.END, text + "\n")

def generate_idea_artifacts():
    text = generate_ideas_text("artifacts")
    notepad_sketchpad.notepad.insert(tk.END, text + "\n")

def generate_idea_activities():
    text = generate_ideas_text("hobbies")
    notepad_sketchpad.notepad.insert(tk.END, text + "\n")

def generate_idea_places():
    text = generate_ideas_text("places")
    notepad_sketchpad.notepad.insert(tk.END, text + "\n")

def generate_idea_balderdash():
    text = generate_ideas_text("balderdash")
    notepad_sketchpad.notepad.insert(tk.END, text + "\n")

def generate_idea_alien_planets():
    text = generate_ideas_text("planets")
    notepad_sketchpad.notepad.insert(tk.END, text + "\n")

def generate_idea_magical_spells():
    text = generate_ideas_text("magic spells")
    notepad_sketchpad.notepad.insert(tk.END, text + "\n")

def generate_idea_time_travel_scenarios():
    text = generate_ideas_text("time travel")
    notepad_sketchpad.notepad.insert(tk.END, text + "\n")

def generate_idea_mythical_creature_encounters():
    text = generate_ideas_text("creatures")
    notepad_sketchpad.notepad.insert(tk.END, text + "\n")

def generate_idea_dream_descriptions():
    text = generate_ideas_text("dreams")
    notepad_sketchpad.notepad.insert(tk.END, text + "\n")

def generate_idea_events():
    text = generate_ideas_text("events")
    notepad_sketchpad.notepad.insert(tk.END, text + "\n")

def generate_idea_lost_key():
    text = generate_ideas_text("lost_key")
    notepad_sketchpad.notepad.insert(tk.END, text + "\n")

def generate_idea_recipe():
    text = generate_ideas_text("recipe")
    notepad_sketchpad.notepad.insert(tk.END, text + "\n")

#----------------------------------------------------------------------------------------------------------------------

# T E X T    P R O M P T S

def generate_ideas_text(request_type):
    request_dict = {
        'movie_scene': 'Make up a very imaginative movie scene.',
        'art_and_photo_styles': 'Invent art and photo styles, separated by comma.',
        'uncommon_adj': 'List 50 uncommon adjectives, separated by comma.',
        'trinkets': 'Give me a list of 50 uncommon trinkets immediately, separated by comma.',
        'artifacts': 'Provide me with 50 uncommon artifacts, separated by comma. Begin the list immediately.',
        'hobbies': 'Provide me with 50 uncommon hobbies, listed by comma.',
        'places': 'Provide me with a unique list of places someone or something could be. Answer immediately.',
        'balderdash': 'Pick a word from the Balderdash dictionary and define it.',
        'dreams': 'Generate a dream where the dreamer becomes a time-traveling sorcerer, exploring fantastical realms and encountering alien creatures while searching for a lost artifact that can alter the fabric of dreams.',
        'planets': 'Imagine a lush alien planet where time flows differently, inhabited by ethereal beings who can cast spells using the energy of bioluminescent flora.',
        'creatures': 'Imagine a fascinating creature from the depths of your imagination. Describe its unique appearance, extraordinary abilities, and the mystical aura it possesses. Bring to life a creature that sparks wonder and captures the imagination of those who encounter it.',
        'time travel': 'Create a time travel adventure where a group of explorers use enchanted artifacts to travel to different alien planets and witness the evolution of their magical ecosystems.',
        'magic spells': 'Invent a powerful spell that allows the caster to open portals to different time periods and explore the mysteries of ancient civilizations.',
        'events': 'create and list 50 odd events in paragraph form, separated by commas',
        'lost_key': 'Describe the journey of a lost key as it encounters different people and objects.',
        'recipe': 'Create a fictional recipe for a dish that can evoke specific emotions in those who eat it.',
        
    }

    request_sentence = request_dict[request_type]

    # Setup variables for ChatCompletion.create call
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": request_sentence},
        {"role": "assistant", "content": "Here ya go! :"}
    ]

    model = "gpt-4"
    max_tokens = 100
    temperature = 0.9

    # Call the GPT-3 model to generate a set of words
    response = openai.ChatCompletion.create(
        model = model,
        messages=conversation,
        max_tokens = max_tokens,
        temperature = temperature
    )

    # Extract the assistant's reply
    assistant_reply = response['choices'][0]['message']['content'].strip()

    # Truncate the response to a maximum of 300 words
    words = assistant_reply.split()
    truncated_reply = ' '.join(words[:300])

    return truncated_reply


#======================================================================================================================
#======================================================================================================================
#======================================================================================================================
#======================================================================================================================
#======================================================================================================================
#======================================================================================================================

root = tk.Tk()
root.title("brAInstormer")

# Maximize the window
root.state('zoomed')

#======================================================================================================================
#======================================================================================================================
#======================================================================================================================
#======================================================================================================================
#======================================================================================================================
#======================================================================================================================


#******* G  U  I ******* ******* G  U  I ******* ******* G  U  I ******* ******* G  U  I ******* ******* G  U  I *******
#******* G  U  I ******* ******* G  U  I ******* ******* G  U  I ******* ******* G  U  I ******* ******* G  U  I *******
#******* G  U  I ******* ******* G  U  I ******* ******* G  U  I ******* ******* G  U  I ******* ******* G  U  I *******


#======================================================================================================================
#======================================================================================================================
#======================================================================================================================
#======================================================================================================================
#======================================================================================================================
#======================================================================================================================


# L O G O

frame = tk.Frame(root)
frame.grid(row=1, column=4, rowspan=3, sticky="nsew", padx=2, pady=2)

text = " brAInstormer"

colors = ["white", "gray"]

text_widget = tk.Text(frame, font=("Courier", 35))
text_widget.pack()

# Insert each character with a different color
for i, char in enumerate(text):
    if char.lower() in ['a', 'i']:
        color_index = 0  # White color for the letters 'a' and 'I'
    else:
        color_index = 1  # Gray color for other characters

    text_widget.tag_configure(f"colored{i}", foreground=colors[color_index])
    text_widget.insert("end", char, f"colored{i}")

# Disable editing and set the height and width to fit the content
text_widget.config(state="disabled", height=1, width=len(text))

#======================================================================================================================
#======================================================================================================================


# I M A G E    G E N E R A T I O N

# Create a new frame to hold the widgets
image_generation_frame = tk.Frame(root, bg='#262626', highlightbackground='dark gray', highlightcolor='dark gray', highlightthickness=1, bd=0)

# Configure columns in the frame to expand
image_generation_frame.columnconfigure(0, weight=1)
image_generation_frame.columnconfigure(1, weight=1)

# Create the widgets within the frame
label_text = tk.Label(image_generation_frame, text="Text to Image:")

# Drop down menu for model selection
model_var = tk.StringVar()
model_choices = {"Stable Diffusion", "DALLE-2"}
model_var.set("DALLE-2")  # set the default option
model_prompt = tk.OptionMenu(image_generation_frame, model_var, *model_choices)

# Entry field for text prompt
text_entry = tk.Entry(image_generation_frame, width=30)

# Step amount label and entry field
steps_label = tk.Label(image_generation_frame, text="Steps:")
steps_entry = tk.Entry(image_generation_frame, width=5)

button_generate = tk.Button(image_generation_frame, text="Generate Image", command=lambda: generate_image_and_save() if model_var.get() == "Stable Diffusion" else generate_image_dalle(), bg='light blue')
button_randomize = tk.Button(image_generation_frame, text="Total Chaos", command=button_randomize_click, bg='light blue')

# Button for the single Random Chaos
button_single_randomize = tk.Button(image_generation_frame, text="Smart Chaos (GPT-4)", command=random_chaos_single, bg='light blue')

# Place the new button in the frame
button_single_randomize.grid(row=3, column=2, sticky='W')

# Place the widgets in the frame
label_text.grid(row=0, column=0, sticky='W', columnspan=2)
model_prompt.grid(row=1, column=0, sticky='W')
text_entry.grid(row=1, column=1, sticky='W', columnspan=2)
steps_label.grid(row=2, column=0, sticky='W')
steps_entry.grid(row=2, column=1, sticky='W')
button_generate.grid(row=3, column=0, sticky='W')
button_randomize.grid(row=3, column=1, sticky='W', columnspan=2)

# Place the frame in the root window
image_generation_frame.grid(row=1, column=1, sticky='W')


#======================================================================================================================
#======================================================================================================================


# O R I G I N A L   I M A G E    G U I

image_frame = tk.Frame(root, bg='#262626', highlightbackground='dark gray', highlightcolor='dark gray', highlightthickness=1, bd=0)
image_label = tk.Label(image_frame)
image_label.grid(row=0, column=0, sticky='nsew')
image_frame.grid(row=2, column=2, padx=0, pady=0, sticky='nsew')


#======================================================================================================================
#======================================================================================================================

# D A L L E   F R A M E

# Create image label in the main window, not in the DALL·E frame
lbl = tk.Label(root, bg='#262626', highlightbackground='dark gray', highlightcolor='dark gray', highlightthickness=1, bd=0)
lbl.grid(row=2, column=2, padx=10, pady=10, sticky='nsew')


#======================================================================================================================
#======================================================================================================================

# F X   I M A G E   D I S P L A Y    G U I

fx_image_frame = tk.Frame(root, bg='#262626', highlightbackground='dark gray', highlightcolor='dark gray', highlightthickness=1, bd=0)
fx_image_label = tk.Label(fx_image_frame)
fx_image_label.pack(fill='both', expand=True)
fx_image_frame.grid(row=2, column=3, padx=10, pady=10, sticky='nsew')


#======================================================================================================================
#======================================================================================================================


# F X    B U T T O N S    G U I

# Create the notebook widget to hold the tabs
notebook = ttk.Notebook(root)
notebook.grid(row=2, column=1, padx=10, pady=10, sticky='W')

# Tab 1: Image Editing
editing_tab = ttk.Frame(notebook)
notebook.add(editing_tab, text='Image Tools')

# Column 1
#upscale_button = tk.Button(editing_tab, text="Upscale", command=upscale_image, bg='black')
remove_background_button = tk.Button(editing_tab, text="Remove Background", command=remove_background, bg='light blue')
mirror_button = tk.Button(editing_tab, text="Mirror", command=mirror_image, bg='black')
flip_vertically_button = tk.Button(editing_tab, text="Flip", command=flip_vertically, bg='black')
rotate_left_button = tk.Button(editing_tab, text="Rotate Left", command=rotate_left, bg='black')
rotate_right_button = tk.Button(editing_tab, text="Rotate Right", command=rotate_right, bg='black')
greyscale_button = tk.Button(editing_tab, text="Greyscale", command=greyscale_image, bg='black')
sepia_button = tk.Button(editing_tab, text="Sepia", command=sepia_tone, bg='black')
auto_crop_button = tk.Button(editing_tab, text="Auto Crop", command=auto_crop, bg='black')

#upscale_button.grid(row=0, column=0, sticky='W')
remove_background_button.grid(row=1, column=0, sticky='W')
mirror_button.grid(row=2, column=0, sticky='W')
flip_vertically_button.grid(row=3, column=0, sticky='W')
rotate_left_button.grid(row=4, column=0, sticky='W')
rotate_right_button.grid(row=5, column=0, sticky='W')
greyscale_button.grid(row=6, column=0, sticky='W')
sepia_button.grid(row=7, column=0, sticky='W')
auto_crop_button.grid(row=7, column=1, sticky='W')

# Add the generate variations button to the DALL·E frame
#variation_button = tk.Button(editing_tab, text="Variation", command=generate_variations)
#variation_button.grid(row=0, column=1, sticky='W')

# Column 2
sharpen_button = tk.Button(editing_tab, text="Sharpen", command=sharpen_image, bg='black')
blur_button = tk.Button(editing_tab, text="Blur", command=blur_image, bg='black')
noise_reduction_button = tk.Button(editing_tab, text="Noise Reduction", command=noise_reduction, bg='black')
saturation_button = tk.Button(editing_tab, text="Saturate", command=enhance_saturation, bg='black')
brighten_button = tk.Button(editing_tab, text="Brighten", command=brighten_image, bg='black')
darken_button = tk.Button(editing_tab, text="Darken", command=darken_image, bg='black')

sharpen_button.grid(row=1, column=1, sticky='W')
blur_button.grid(row=2, column=1, sticky='W')
noise_reduction_button.grid(row=3, column=1, sticky='W')
saturation_button.grid(row=4, column=1, sticky='W')
brighten_button.grid(row=5, column=1, sticky='W')
darken_button.grid(row=6, column=1, sticky='W')

#Tab 2: Image Filters
effects_tab = ttk.Frame(notebook)
notebook.add(effects_tab, text='Image Filters')

# Create and place the Image Filters buttons within the Effects tab
emboss_button = tk.Button(effects_tab, text="Emboss", command=emboss_image, bg='black')
solarize_button = tk.Button(effects_tab, text="Solarize", command=solarize_image, bg='black')
posterize_button = tk.Button(effects_tab, text="Posterize", command=posterize_image, bg='black')
pixelate_button = tk.Button(effects_tab, text="Pixelate", command=pixelate_image, bg='black')
vhs_glitch_button = tk.Button(effects_tab, text="VHS Glitch", command=apply_vhs_glitch_effect, bg='black')
oil_painting_button = tk.Button(effects_tab, text="Oil Painting", command=apply_oil_painting_effect, bg='black')
vignette_button = tk.Button(effects_tab, text="Vignette", command=vignette, bg='black')

emboss_button.grid(row=1, column=1, sticky='W')
solarize_button.grid(row=3, column=0, sticky='W')
posterize_button.grid(row=4, column=0, sticky='W')
pixelate_button.grid(row=5, column=0, sticky='W')
vhs_glitch_button.grid(row=6, column=0, sticky='W')
oil_painting_button.grid(row=7, column=0, sticky='W')
vignette_button.grid(row=2, column=1, sticky='W')

# Dreamy Effect Button
dreamy_effect_button = tk.Button(effects_tab, text="Dreamy ", command=dreamy_effect, bg='black')
dreamy_effect_button.grid(row=0, column=0, sticky='W')

# Grainy Art Button
glitch_art_button = tk.Button(effects_tab, text="Grainy", command=glitch_art, bg='black')
glitch_art_button.grid(row=1, column=0, sticky='W')

# New "Thermal Vision" button
thermal_vision_button = tk.Button(effects_tab, text="Thermal Vision", command=thermal_vision, bg='black')
thermal_vision_button.grid(row=3, column=1, sticky='W')

# Edge Detection
edge_detection_button = tk.Button(effects_tab, text="Edge Detection", command=edge_detection, bg='black')
edge_detection_button.grid(row=0, column=1, sticky='W')

# Vintage
vintage_button = tk.Button(effects_tab, text="Vintage", command=vintage_filter, bg='black')
vintage_button.grid(row=2, column=0, sticky='W')

# Cartoonize
cartoonize_button = tk.Button(effects_tab, text="Cartoonize", command=cartoonize, bg='black')
cartoonize_button.grid(row=4, column=1, sticky='W')

# Abstract Button
pop_culture_button = tk.Button(effects_tab, text="Abstract", command=pop_culture_filter, bg='black')
pop_culture_button.grid(row=5, column=1, columnspan=2, sticky='W')

# invert
invert_button = tk.Button(effects_tab, text="Invert Colors", command=invert_colors, bg='black')
invert_button.grid(row=6, column=1, sticky='W')  # Updated grid position

# Pencil Sketch
pencil_sketch_button = tk.Button(effects_tab, text="Pencil Sketch", command=pencil_sketch, bg='black')
pencil_sketch_button.grid(row=7, column=1, sticky='W')

#----------------------------------------------------------------------------------------------------------------------

# A D D   F O N T

# Add Text to Image Tab
text_to_image_tab = ttk.Frame(notebook)
notebook.add(text_to_image_tab, text='Add Text')

# Create and place the Add Text to Image parameters within the Text to Image tab
label_text_on_image = tk.Label(text_to_image_tab, text="Text:")
text_on_image_entry = tk.Entry(text_to_image_tab, width=15)

label_font = tk.Label(text_to_image_tab, text="Font:")
fonts = ["Arial", "Courier", "Helvetica", "Times", "Verdana"]
font_var = tk.StringVar(text_to_image_tab)
font_var.set(fonts[0])  # set default value
font_optionmenu = tk.OptionMenu(text_to_image_tab, font_var, *fonts)

label_font_size = tk.Label(text_to_image_tab, text="Font Size:")
font_sizes = list(range(10, 101, 10))
font_size_var = tk.StringVar(text_to_image_tab)
font_size_var.set(font_sizes[0])  # set default value
font_size_optionmenu = tk.OptionMenu(text_to_image_tab, font_size_var, *font_sizes)

label_color = tk.Label(text_to_image_tab, text="Text Color:")
colors = ["black", "white", "red", "green", "blue", "yellow", "pink", "orange"]
color_var = tk.StringVar(text_to_image_tab)
color_var.set(colors[0])  # set default value
color_optionmenu = tk.OptionMenu(text_to_image_tab, color_var, *colors)

label_position = tk.Label(text_to_image_tab, text="Position:")
x_position_var = tk.StringVar(text_to_image_tab)
y_position_var = tk.StringVar(text_to_image_tab)
position_frame = tk.Frame(text_to_image_tab)
label_x_position = tk.Label(position_frame, text="X:")
x_position_entry = tk.Entry(position_frame, textvariable=x_position_var, width=5)
label_y_position = tk.Label(position_frame, text="Y:")
y_position_entry = tk.Entry(position_frame, textvariable=y_position_var, width=5)
label_text_on_image.grid(row=0, column=0, sticky='W', padx=10, pady=5)
text_on_image_entry.grid(row=0, column=1, sticky='W', padx=10, pady=5)
label_font.grid(row=1, column=0, sticky='W', padx=10, pady=5)
font_optionmenu.grid(row=1, column=1, sticky='W', padx=10, pady=5)
label_font_size.grid(row=2, column=0, sticky='W', padx=10, pady=5)
font_size_optionmenu.grid(row=2, column=1, sticky='W', padx=10, pady=5)
label_color.grid(row=3, column=0, sticky='W', padx=10, pady=5)
color_optionmenu.grid(row=3, column=1, sticky='W', padx=10, pady=5)
label_position.grid(row=4, column=0, sticky='W', padx=10, pady=5)
position_frame.grid(row=4, column=1, sticky='W', padx=10, pady=5)
label_x_position.grid(row=0, column=0, sticky='W')
x_position_entry.grid(row=0, column=1, sticky='W')
label_y_position.grid(row=0, column=2, sticky='W')
y_position_entry.grid(row=0, column=3, sticky='W')
button_add_text = tk.Button(text_to_image_tab, text="Add Text", command=add_text_on_image, bg='light blue')
button_add_text.grid(row=5, column=0, columnspan=2, sticky='W', padx=10, pady=5)


#======================================================================================================================
#======================================================================================================================


# I M A G E   T O   T O   I M A G E   G U I

# Create a new frame to hold the Image-to-Image settings
image_to_image_frame = tk.Frame(root, bg='#262626', highlightbackground='dark gray', highlightcolor='dark gray', highlightthickness=1, bd=0)

# Create the label and text entry field within the frame
image_to_image_label_text = tk.Label(image_to_image_frame, text="Image to Image:")
image_to_image_entry = tk.Entry(image_to_image_frame, width=31)

# Create the Image-to-Image button
button_image_to_image = tk.Button(image_to_image_frame, text="Generate Img-2-Img", command=image_to_image, bg='light blue')

# Create the button to upload image for Image-to-Image
button_upload_for_itf = tk.Button(image_to_image_frame, text="Import Image", command=upload_image_for_itf, bg='light blue')

# Create the button to generate image variations
variation_button = tk.Button(image_to_image_frame, text="Generate Variation", command=generate_variations, bg='light blue')

# Create the Upscale button
button_upscale = tk.Button(image_to_image_frame, text="Upscale Image", command=upscale_image, bg='light blue')

# Place the widgets in the frame using grid
image_to_image_label_text.grid(row=0, column=0, columnspan=2)  # spans two columns
image_to_image_entry.grid(row=1, column=0, columnspan=2)  # spans two columns
button_upload_for_itf.grid(row=2, column=0)  # placed in the left column
button_image_to_image.grid(row=2, column=1)  # placed in the right column
variation_button.grid(row=3, column=1)  # placed below import button
button_upscale.grid(row=3, column=0)  # placed below generate button

# Add the generate variations button to the DALL·E frame
#variation_button = tk.Button(editing_tab, text="Variation", command=generate_variations)
#variation_button.grid(row=0, column=1, sticky='W')

# Place the frame in the root window
image_to_image_frame.grid(row=1, column=2, sticky='W')

# IMAGE DISPLAY FRAME
image_display_frame = tk.Frame(root, bg='#262626', highlightbackground='dark gray', highlightcolor='dark gray', highlightthickness=1, bd=0)

# Create a label to display the uploaded image
lbl = tk.Label(image_display_frame)
lbl.pack()

# Place the frame in the root window
image_display_frame.grid(row=2, column=2, sticky='W')

#======================================================================================================================
#======================================================================================================================


# N O T E    A N D   S K E T C H P A D    G U I

# Create a Notebook for tabs
notebook = ttk.Notebook(root)
notebook.grid(row=3, column=3, sticky='nsew')

notepad_sketchpad = NotepadSketchpad(root, width=100, height=100)
sketchpad = Sketchpad(root, width=100, height=100)

notebook.add(notepad_sketchpad, text='Notepad')
notebook.add(sketchpad, text='Sketchpad')

# Buttons frame with grey border
buttons_frame = tk.Frame(root, highlightthickness=1, highlightbackground="grey", highlightcolor="grey")
buttons_frame.grid(row=4, column=3, sticky='n', pady=(0, 0))

toolbar = tk.Frame(buttons_frame)
toolbar.grid(row=0, column=0, columnspan=10, sticky='nsew')

italic_btn = tk.Button(toolbar, text='Italic', command=notepad_sketchpad.toggle_italic, width=3, font=("Helvetica", 11))
italic_btn.grid(row=0, column=0,padx=0, pady=0)

bold_btn = tk.Button(toolbar, text='Bold', command=notepad_sketchpad.toggle_bold, width=3, font=("Helvetica", 11))
bold_btn.grid(row=0, column=1, padx=0, pady=0)
idea_var = tk.StringVar()
idea_var.set("Scenes")

idea_menu = ttk.OptionMenu(toolbar, idea_var, "Scenes", "Styles", "Adjectives", "Trinkets", "Artifacts", "Activities",
                           "Places", "Balderdash", "Planets", "Magic", "Time Travel", "Creatures", "Dreams", "Events", "Lost Key", "Recipe",)
idea_menu.grid(row=1, column=2, padx=0, pady=0)

def generate_idea():
    selected_idea = idea_var.get()
    if selected_idea == "Scenes":
        generate_idea_scenes()
    elif selected_idea == "Styles":
        generate_idea_styles()
    elif selected_idea == "Adjectives":
        generate_idea_adjectives()
    elif selected_idea == "Trinkets":
        generate_idea_trinkets()
    elif selected_idea == "Artifacts":
        generate_idea_artifacts()
    elif selected_idea == "Activities":
        generate_idea_activities()
    elif selected_idea == "Places":
        generate_idea_places()
    elif selected_idea == "Balderdash":
        generate_idea_balderdash()
    elif selected_idea == "Planets":
        generate_idea_alien_planets()
    elif selected_idea == "Magic":
        generate_idea_magical_spells()
    elif selected_idea == "Time Travel":
        generate_idea_time_travel_scenarios()
    elif selected_idea == "Creatures":
        generate_idea_mythical_creature_encounters()
    elif selected_idea == "Dreams":
        generate_idea_dream_descriptions()
    elif selected_idea == "Events":
        generate_idea_events()
    elif selected_idea == "Lost Key":
        generate_idea_lost_key()
    elif selected_idea == "Recipe":
        generate_idea_recipe()
    
def change_both_colors():
    color = colorchooser.askcolor()[1]
    if color:
        notepad_sketchpad.select_color()
        sketchpad.select_color()

generate_idea_btn = tk.Button(toolbar, text="Generate Idea", command=generate_idea, width=9, font=("Helvetica", 11))
generate_idea_btn.grid(row=1, column=3, padx=0, pady=0)

clear_sketch_btn = tk.Button(toolbar, text='Clear Art', command=sketchpad.clear_sketch, width=6, font=("Helvetica", 11))
clear_sketch_btn.grid(row=0, column=3, padx=0, pady=0)

save_sketch_btn = tk.Button(toolbar, text='Save Art', command=sketchpad.save_sketch_as_jpg, width=6, font=("Helvetica", 11))
save_sketch_btn.grid(row=1, column=0, padx=0, pady=0)

button_save_note = tk.Button(toolbar, text="Save Txt", command=notepad_sketchpad.save_notepad_contents, width=6, font=("Helvetica", 11))
button_save_note.grid(row=1, column=1, padx=0, pady=0)

color_change_btn = tk.Button(toolbar, text="Color", command=change_both_colors, width=5, font=("Helvetica", 11))
color_change_btn.grid(row=0, column=2, padx=0, pady=0)

scrollbar = tk.Scrollbar(notepad_sketchpad)
scrollbar.pack(side='right', fill='y')

    
#======================================================================================================================
#======================================================================================================================


# E X P E R I M E N T A L   C O N V E R S I O N S    G U I


conversion_frame = tk.Frame(root, bg='#262626', highlightbackground='dark gray', highlightcolor='dark gray', highlightthickness=1, bd=0)
conversion_frame.grid(row=1, column=3, sticky='nsew')

# Create the label and text entry field within the frame
experimental_label = tk.Label(conversion_frame, text="Experimental Conversions:")
experimental_label.grid(row=0, column=0, columnspan=2, sticky='n')

# Buttons for generating MIDI from image (row 0)
button_generate_midi_a = tk.Button(conversion_frame, text="Image->MIDI #1", command=generate_midi_from_image, bg='light blue')
button_generate_midi_a.grid(row=1, column=0)

button_generate_audio_a = tk.Button(conversion_frame, text="Image->Audio #1", command=generate_audio_from_image, bg='light blue')
button_generate_audio_a.grid(row=1, column=1)

# Buttons for generating MIDI from image (row 1)
button_generate_midi_b = tk.Button(conversion_frame, text="Image->MIDI #2", command=generate_midi_from_image, bg='light blue')
button_generate_midi_b.grid(row=2, column=0)

button_generate_audio_b = tk.Button(conversion_frame, text="Image->Audio #2", command=generate_audio_from_image, bg='light blue')
button_generate_audio_b.grid(row=2, column=1)

button_image_to_ascii = tk.Button(conversion_frame, text="Image->ASCII", command=convert_image_to_ascii, bg='light blue')
button_image_to_ascii.grid(row=3, column=0)

import_to_ascii_button = tk.Button(conversion_frame, text="Import->ASCII", command=import_img_to_ascii, bg='light blue')
import_to_ascii_button.grid(row=3, column=1)

# Configure the column to center the audio frame
conversion_frame.grid_columnconfigure(0, weight=1)
conversion_frame.grid_columnconfigure(1, weight=1)


#======================================================================================================================
#======================================================================================================================


# A N I M A T I O N     P A R A M E T E R S    G U I (Text to Animate tab)

notebook = ttk.Notebook(root)
notebook.grid(row=3, column=1, columnspan=1, padx=1, pady=1)

# Create the frames for each tab
text_to_animate_frame = ttk.Frame(notebook)
image_to_animate_frame = ttk.Frame(notebook)

# Add the frames to the notebook with respective titles
notebook.add(text_to_animate_frame, text="Text to Animate")
notebook.add(image_to_animate_frame, text="Image to Animate")

# Buttons and text frame
animation_frame_text = tk.Frame(text_to_animate_frame, highlightbackground='gray', highlightcolor='gray', highlightthickness=1, bd=0)

# Create the labels, entry fields, and buttons within the frame
start_label_text = tk.Label(animation_frame_text, text="Start Prompt:")
animation_start_entry = tk.Entry(animation_frame_text, width=14)
start_frame_label_text = tk.Label(animation_frame_text, text="Start at 0:")
start_frame_entry = tk.Entry(animation_frame_text, width=3)

mid_label_text = tk.Label(animation_frame_text, text="Mid 1 Prompt:")
animation_mid_entry = tk.Entry(animation_frame_text, width=14)
mid_frame_label_text = tk.Label(animation_frame_text, text="Mid 1 Frame:")
mid_frame_entry = tk.Entry(animation_frame_text, width=3)

mid_label_text2 = tk.Label(animation_frame_text, text="Mid 2 Prompt:")
animation_mid_entry2 = tk.Entry(animation_frame_text, width=14)
mid_frame_label_text2 = tk.Label(animation_frame_text, text="Mid 2 Frame:")
mid_frame_entry2 = tk.Entry(animation_frame_text, width=3)

mid_label_text3 = tk.Label(animation_frame_text, text="Mid 3 Prompt:")
animation_mid_entry3 = tk.Entry(animation_frame_text, width=14)
mid_frame_label_text3 = tk.Label(animation_frame_text, text="Mid 3 Frame:")
mid_frame_entry3 = tk.Entry(animation_frame_text, width=3)

end_label_text = tk.Label(animation_frame_text, text="End Prompt:")
animation_end_entry = tk.Entry(animation_frame_text, width=14)
end_frame_label_text = tk.Label(animation_frame_text, text="End Frame:")
end_frame_entry = tk.Entry(animation_frame_text, width=3)

length_label_text = tk.Label(animation_frame_text, text="Total Frames:")
animation_length_entry = tk.Entry(animation_frame_text, width=3)

seed_label_text = tk.Label(animation_frame_text, text="Seeds:")
seed_entry = tk.Entry(animation_frame_text, width=3)

fps_label_text = tk.Label(animation_frame_text, text="FPS:")
fps_entry = tk.Entry(animation_frame_text, width=3)

button_generate_video = tk.Button(animation_frame_text, text="Generate Animation", command=generate_video, bg='#4CAF50', relief="raised", bd=0, padx=5)

# Place the widgets in the frame using grid
start_label_text.grid(row=1, column=0, sticky='W', padx=(0, 0), pady=(0, 0))
animation_start_entry.grid(row=1, column=1, sticky='W', padx=(0, 0), pady=(0, 0))
start_frame_label_text.grid(row=1, column=2, sticky='W', padx=(0, 0), pady=(0, 0))
start_frame_entry.grid(row=1, column=3, sticky='W', padx=(0, 0), pady=(0, 0))

mid_label_text.grid(row=2, column=0, sticky='W', padx=(0, 0), pady=(0, 0))
animation_mid_entry.grid(row=2, column=1, sticky='W', padx=(0, 0), pady=(0, 0))
mid_frame_label_text.grid(row=2, column=2, sticky='W', padx=(0, 0), pady=(0, 0))
mid_frame_entry.grid(row=2, column=3, sticky='W', padx=(0, 0), pady=(0, 0))

mid_label_text2.grid(row=3, column=0, sticky='W', padx=(0, 0), pady=(0, 0))
animation_mid_entry2.grid(row=3, column=1, sticky='W', padx=(0, 0), pady=(0, 0))
mid_frame_label_text2.grid(row=3, column=2, sticky='W', padx=(0, 0), pady=(0, 0))
mid_frame_entry2.grid(row=3, column=3, sticky='W', padx=(0, 0), pady=(0, 0))

mid_label_text3.grid(row=4, column=0, sticky='W', padx=(0, 0), pady=(0, 0))
animation_mid_entry3.grid(row=4, column=1, sticky='W', padx=(0, 0), pady=(0, 0))
mid_frame_label_text3.grid(row=4, column=2, sticky='W',padx=(0, 0), pady=(0, 0))
mid_frame_entry3.grid(row=4, column=3, sticky='W', padx=(0, 0), pady=(0, 0))

end_label_text.grid(row=5, column=0, sticky='W', padx=(0, 0), pady=(0, 0))
animation_end_entry.grid(row=5, column=1, sticky='W', padx=(0, 0), pady=(0, 0))
end_frame_label_text.grid(row=5, column=2, sticky='W', padx=(0, 0), pady=(0, 0))
end_frame_entry.grid(row=5, column=3, sticky='W', padx=(0, 0), pady=(0, 0))

length_label_text.grid(row=6, column=2, sticky='W', padx=(0, 0), pady=(0, 0))
animation_length_entry.grid(row=6, column=3, sticky='W', padx=(0, 0), pady=(0, 0))

seed_label_text.grid(row=7, column=0, sticky='W', padx=(0, 0), pady=(0, 0))
seed_entry.grid(row=7, column=1, sticky='W', padx=(0, 0), pady=(0, 0))

fps_label_text.grid(row=6, column=0, sticky='W', padx=(0, 0), pady=(0, 0))
fps_entry.grid(row=6, column=1, sticky='W', padx=(0, 0), pady=(0, 0))

button_generate_video.grid(row=10, column=0, columnspan=4, padx=0, pady=(0, 0))

# Create checkboxes for each feature
zoom_state = tk.IntVar()
zoom_checkbox = tk.Checkbutton(animation_frame_text, text="Zoom", variable=zoom_state)
zoom_checkbox.grid(row=7, column=2, sticky='W', padx=(0, 0), pady=(0, 0))

up_state = tk.IntVar()
up_checkbox = tk.Checkbutton(animation_frame_text, text="Up", variable=up_state)
up_checkbox.grid(row=8, column=0, sticky='W', padx=(0, 0), pady=(0, 0))

down_state = tk.IntVar()
down_checkbox = tk.Checkbutton(animation_frame_text, text="Down", variable=down_state)
down_checkbox.grid(row=8, column=1, sticky='W', padx=(0, 0), pady=(0, 0))

left_state = tk.IntVar()
left_checkbox = tk.Checkbutton(animation_frame_text, text="Left", variable=left_state)
left_checkbox.grid(row=8, column=2, sticky='W', padx=(0, 0), pady=(0, 0))

right_state = tk.IntVar()
right_checkbox = tk.Checkbutton(animation_frame_text, text="Right", variable=right_state)
right_checkbox.grid(row=8, column=3, sticky='W', padx=(0, 0), pady=(0, 0))

# Define preset options
preset_options = ["None", "3d-model", "analog-film", "anime", "cinematic", "comic-book", "digital-art",
                  "enhance", "fantasy-art", "isometric", "line-art", "low-poly", "modeling-compound",
                  "neon-punk", "origami", "photographic", "pixel-art"]

# Create a label for the preset selection
preset_label = tk.Label(animation_frame_text, text="Style:")
preset_label.grid(row=9, column=0, sticky='W', padx=(0, 0), pady=(0, 0))

# Create the Combobox for the preset selection
preset_combobox = ttk.Combobox(animation_frame_text, values=preset_options, width=10)
preset_combobox.grid(row=9, column=1, columnspan=1, sticky='W', padx=(0, 0), pady=(0, 0))

# Place the frame in the root window
animation_frame_text.grid(row=0, column=0, padx=0, pady=0, sticky='W')


#======================================================================================================================
#======================================================================================================================


# I M A G E   T O   A N I M A T E   G U I 

# Creating the frame for Image to Animate
animation_frame_image = tk.Frame(image_to_animate_frame)

# Create a Frame to hold the image label
image_frame = tk.Frame(animation_frame_image, width=100, height=100, bd=1, relief='solid')
image_frame.grid(row=0, column=1)

# Make sure the frame does not resize to fit the label by disabling resizing on both axes
image_frame.grid_propagate(False)

# Add the label to this frame instead of directly to the parent frame
img = tk.Label(image_frame)
img.grid()

# Import video button
import_button = tk.Button(animation_frame_image, text="Import", command=import_video, width=7)
import_button.grid(row=0, column=0, pady=(0, 0), padx=(0, 0))

# Create the labels and entry fields
start_image_label = tk.Label(animation_frame_image, text="Start Prompt:")
start_image_entry = tk.Entry(animation_frame_image, width=13)
start_frame_label_image = tk.Label(animation_frame_image, text="Start at 0:")
start_frame_entry_image = tk.Entry(animation_frame_image, width=3)

mid_image_label = tk.Label(animation_frame_image, text="Mid 1 Prompt:")
mid_image_entry = tk.Entry(animation_frame_image, width=13)
mid_frame_label_image = tk.Label(animation_frame_image, text="Mid 1 Frame:")
mid_frame_entry_image = tk.Entry(animation_frame_image, width=3)

mid_image_label2 = tk.Label(animation_frame_image, text="Mid 2 Prompt:")
mid_image_entry2 = tk.Entry(animation_frame_image, width=13)
mid_frame_label_image2 = tk.Label(animation_frame_image, text="Mid 2 Frame:")
mid_frame_entry_image2 = tk.Entry(animation_frame_image, width=3)

mid_image_label3 = tk.Label(animation_frame_image, text="Mid 3 Prompt:")
mid_image_entry3 = tk.Entry(animation_frame_image, width=13)
mid_frame_label_image3 = tk.Label(animation_frame_image, text="Mid 3 Frame:")
mid_frame_entry_image3 = tk.Entry(animation_frame_image, width=3)

end_image_label = tk.Label(animation_frame_image, text="End Prompt:")
end_image_entry = tk.Entry(animation_frame_image, width=13)
end_frame_label_image = tk.Label(animation_frame_image, text="End at 48:")
end_frame_entry_image = tk.Entry(animation_frame_image, width=3)

# Create the Combobox for the preset selection
preset_label_image = tk.Label(animation_frame_image, text="Preset:")
preset_combobox_image = ttk.Combobox(animation_frame_image, values=preset_options, width=10)

# Place all the widgets in the frame using grid
start_image_label.grid(row=1, column=0, sticky='W')
start_image_entry.grid(row=1, column=1, sticky='W')
start_frame_label_image.grid(row=1, column=2, sticky='W')
start_frame_entry_image.grid(row=1, column=3, sticky='W')

mid_image_label.grid(row=2, column=0, sticky='W')
mid_image_entry.grid(row=2, column=1, sticky='W')
mid_frame_label_image.grid(row=2, column=2, sticky='W')
mid_frame_entry_image.grid(row=2, column=3, sticky='W')

mid_image_label2.grid(row=3, column=0, sticky='W')
mid_image_entry2.grid(row=3, column=1, sticky='W')
mid_frame_label_image2.grid(row=3, column=2, sticky='W')
mid_frame_entry_image2.grid(row=3, column=3, sticky='W')

mid_image_label3.grid(row=4, column=0, sticky='W')
mid_image_entry3.grid(row=4, column=1, sticky='W')
mid_frame_label_image3.grid(row=4, column=2, sticky='W')
mid_frame_entry_image3.grid(row=4, column=3, sticky='W')

end_image_label.grid(row=5, column=0, sticky='W')
end_image_entry.grid(row=5, column=1, sticky='W')
end_frame_label_image.grid(row=5, column=2, sticky='W')
end_frame_entry_image.grid(row=5, column=3, sticky='W')

preset_label_image.grid(row=6, column=0, sticky='W')
preset_combobox_image.grid(row=6, column=1, sticky='W')

# Generate image to video button
generate_button = tk.Button(animation_frame_image, text="Generate", command=generate_image_to_video, width=7)
generate_button.grid(row=6, column=2, pady=(0, 0), padx=(0, 0))

# Place the frame in the root window
animation_frame_image.grid(row=0, column=0, padx=0, pady=0, sticky='W')


#======================================================================================================================
#======================================================================================================================


# V I D E O   P L A Y B A C K    G U I

# Create an outer frame with a dark gray background
outer_frame = tk.Frame(root, bd=3, bg='dark gray')
outer_frame.grid(row=3, column=2)

# Calculate the new dimensions to fit within 275x275
new_width = 256
new_height = 256

# Create a frame for the video player inside the outer frame with updated dimensions
video_player_frame = tk.Frame(outer_frame, bd=1, relief="groove", width=new_width, height=new_height)
video_player_frame.pack()

# Initialize filepath with a default video
filepath = 'video.mp4'
video = cv2.VideoCapture(filepath)

# Create a Canvas for the video with a thin off-white border and updated dimensions
canvas = tk.Canvas(video_player_frame, width=new_width, height=new_height, bd=1, highlightbackground='white')
canvas.pack()

# Function to update the video frame on the canvas
def update_video():
    global photo

    # Read the next frame from the video
    ret, frame = video.read()

    if ret:
        # Resize the frame to fit the display frame
        resized_frame = cv2.resize(frame, (new_width, new_height))

        # Convert the frame to RGB format
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

        # Create a PIL ImageTk object from the resized frame
        image = Image.fromarray(rgb_frame)
        photo = ImageTk.PhotoImage(image)

        # Update the canvas with the new frame
        canvas.create_image(0, 0, anchor='nw', image=photo)

    # Schedule the next update after a delay (e.g., 30 milliseconds)
    root.after(30, update_video)

# Call the update_video function to start displaying the video
update_video()

# Create a frame for the buttons
button_frame = tk.Frame(video_player_frame)
button_frame.pack()

# Create control buttons
upload_button = tk.Button(button_frame, text='import', command=upload_video)
upload_button.pack(side=tk.LEFT)
play_button = tk.Button(button_frame, text='play', command=play_video)
play_button.pack(side=tk.LEFT)
pause_button = tk.Button(button_frame, text='pause', command=pause_video)
pause_button.pack(side=tk.LEFT)
stop_button = tk.Button(button_frame, text='stop', command=stop_video)
stop_button.pack(side=tk.LEFT)

# Create a frame for the animation buttons
animation_button_frame = tk.Frame(root, bg='#262626', highlightbackground='dark gray', highlightcolor='dark gray', highlightthickness=1, bd=0)
animation_button_frame.grid(row=4, column=2, pady=10)

# Disable resizing of the root window
root.resizable(False, False)


#======================================================================================================================
#======================================================================================================================

# V I D E O    C H A O S    F R A M E   G U I

video_chaos_button = tk.Button(animation_button_frame, text="Total Video Chaos", command=video_chaos_button_click, bg='light blue')
video_chaos_button.grid(row=0, column=0)  # Adjust grid parameters as needed

random_chaos_button = tk.Button(animation_button_frame, text="Smart Video Chaos (GPT-4)", command=random_chaos)
random_chaos_button.grid(row=0, column=1)  # Adjust grid parameters as needed

animation_button_frame.grid(row=4, column=1)


#======================================================================================================================
#======================================================================================================================


# C H A T   G P T   G U I

def change_tab_name(tab_index):
    new_name = tab_name_vars[tab_index].get()
    if len(new_name) <= 6:
        notebook.tab(tab_index, text=new_name)
    else:
        print("Name is too long. Max 6 characters allowed.")

notebook = ttk.Notebook(root)
notebook.grid(row=2, column=4, rowspan=3, sticky="nsew", padx=2, pady=2)

chat_frames = []
chat_bots = ["Chat1", "Chat2", "Chat3"]
tab_name_vars = []

for bot in chat_bots:
    chat_frame = tk.Frame(notebook, bd=1, relief=tk.RAISED)
    chat_frame.columnconfigure(0, weight=1)
    chat_frame.columnconfigure(1, weight=1)
    chat_frame.rowconfigure(0, weight=1)
    chat_frames.append(chat_frame)
    tab_name_vars.append(tk.StringVar())

for i, chat_frame in enumerate(chat_frames):
    notebook.add(chat_frame, text=chat_bots[i])

    tab_name_entry = tk.Entry(chat_frame, textvariable=tab_name_vars[i], width=5)
    tab_name_entry.grid(row=4, column=0, sticky="nsew", padx=2, pady=2)

    tab_name_button = tk.Button(chat_frame, text="Change Name", command=lambda i=i: change_tab_name(i))
    tab_name_button.grid(row=4, column=1, sticky="nsew", padx=2, pady=2)

notebook.select(chat_frames[0])

chat_logs = []
for chat_frame in chat_frames:
    chat_log = tk.Text(chat_frame, bd=0, bg="#050505", height="20", width="35", font="Arial", wrap=tk.WORD)
    chat_log.tag_config("user_name", foreground="orangered", font=("Arial", 14, "bold")) # user_name in orangered color
    chat_log.tag_config("bot_name", foreground="cyan", font=("Arial", 14, "bold")) # bot_name in lightgreen color
    chat_log.tag_config("user_answer", font=("Arial", 14))
    chat_log.tag_config("bot_answer", font=("Arial", 14))
    chat_log.grid(row=0, column=0, columnspan=2, sticky="nsew")
    chat_log.config(state=tk.DISABLED)
    chat_logs.append(chat_log)

scrollbars = []
for chat_frame, chat_log in zip(chat_frames, chat_logs):
    scrollbar = tk.Scrollbar(chat_frame, command=chat_log.yview)
    scrollbar.grid(row=0, column=2, sticky="ns")
    chat_log.config(yscrollcommand=scrollbar.set)
    scrollbars.append(scrollbar)

user_inputs = []
for chat_frame in chat_frames:
    user_input = tk.Entry(chat_frame, bd=0, bg="#050505", width="20", font="Arial")
    user_input.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=2, pady=2)
    user_inputs.append(user_input)

send_buttons = []
for chat_frame in chat_frames:
    send_button = tk.Button(chat_frame, text="Send Message", bg="#050505", command=get_chat_response)
    send_button.grid(row=2, column=0, sticky="nsew", padx=2, pady=2)
    send_buttons.append(send_button)

# Create clear button for each chatbot conversation
clear_buttons = []
for i, chat_frame in enumerate(chat_frames):
    clear_button = tk.Button(chat_frame, text="Clear Convo", bg="#000505", command=lambda i=i: clear_specific_conversation(i))
    clear_button.grid(row=2, column=1, sticky="nsew", padx=2, pady=2)
    clear_buttons.append(clear_button)

save_buttons = []
for i, chat_frame in enumerate(chat_frames):
    save_button = tk.Button(chat_frame, text="Save Convo", bg="#050505", command=lambda i=i: save_specific_conversation(i))
    save_button.grid(row=3, column=0, sticky="nsew", padx=2, pady=2)
    save_buttons.append(save_button)

load_buttons = []
for i, chat_frame in enumerate(chat_frames):
    load_button = tk.Button(chat_frame, text="Load Convo", bg="#050505", command=lambda i=i: load_specific_conversation(i))
    load_button.grid(row=3, column=1, sticky="nsew", padx=2, pady=2)
    load_buttons.append(load_button)

conversations = [[] for _ in chat_frames]


#======================================================================================================================
#======================================================================================================================
#======================================================================================================================
#======================================================================================================================
#======================================================================================================================


root.mainloop()
