## brAInstormer

brAInstormer is a robust and multifaceted creative suite that leverages the 
power of GPT, Stable Diffusion, Dalle2, and other forms of AI in order to provide 
a diverse set of functionalities from a single screen. 

The core script features a simple but powerful GUI that enables 
capabilities such as text-to-image generation, text-to-animation generation, 
experimental file conversions, a notepad, ChatGPT, and much more, with everything fitting on one screen.

It is also packed with various randomization features powered by GPT, making 
it a one-stop tool to inspire your creativity and streamline concept design.

## Main Features
* Dalle-2 AND Stable Diffusion text-to-image generation.
* Edit, filter, and add font to images with 30+ visual effects that include pixelate, solarize, vignette, greyscale, sepia, mirror, flip, VHS glitch, and many more. 
* Upscale Images: Enhance the quality of any imported image and enlarge it to 2048x2048.
* Image-to-Image generation: Import any image then transform it using text prompts to guide the generation process.
* Image variations: Create endless versions of your generations with the click of a button.
* Text-to-Animation: Generate animations using text prompts, fully equipped with camera and frame controls, 10+ style presets, randomization tools powered by GPT, and more.
* Image-to-Animation: Import any image and use it as a starting prompt to shape 48-frame animations.
* ChatGPT-4: Host up to 3 simultaneous conversations with ChatGPT. Name and assign them different roles and bounce between each chat seamlessly. All conversations can be quickly saved, cleared and imported at any point. 
* Experimental file conversion: Convert images to audio, MIDI, and more.
* Notepad: A convenient tool for note-taking, copy-pasting, or anything else you might need it for. The notebook is also fully equipped with 10+ GPT-powered concept shufflers that can randomly provide movie scene suggestions, art styles, trinket lists, and more.
* Sketchpad: A simple sketchpad for concept doodles with full color spectrum control. Sketches can be saved and cleared at any point. 
* Multiple GPT-powered randomization algorithms to eliminate your creative block.
* All of the above on a single screen means no more tab jumping or app switching. In fact, you don't even need to scroll. 
# Screenshots 
![Software Screenshot 1](/screenshot1.jpeg)

![Software Screenshot 2](/screenshot2.jpeg)

## Set Up

1. Obtain Stability AI and OpenAI API keys. Register on their websites, check their pricing details, and get your own personal API keys:
    - https://platform.openai.com
    - https://dreamstudio.ai/generate

2. Install Python 3: https://www.python.org/downloads/

3. Download `brAInstormer.py`, `words.py`, and `video.mp4` from this repository

4. Create a single folder and put all the downloaded files in it. All generations will be saved in this folder.

5. Install necessary 3rd party modules:
   ```sh
   pip3 install opencv-python
   pip3 install stability-sdk
   pip3 install "stability-sdk[anim]"
   pip3 install tqdm
   pip3 install openai
   pip3 install pydub
   pip3 install midiutil
   ```

6. Download and install ffmpeg: https://ffmpeg.org

7. Open `brainstormer.py` with your Python script editor. Replace 'YOUR API KEY GOES HERE' with your personal API keys in the appropriate places of the script.

8. Run the module and enjoy brainstorming!

## Important
- ***All generations auto-save as the name of the initial prompt in the root folder.***
- I'm brand new to coding. I began self-teaching 6 months ago with the help of ChatGPT in order to build this. So this whole script is the work of a freshman. For this reason, all feedback is absolutely welcomed and encouraged. Just please be kind :) I'd like to make this as awesome as possible.
- This has been tested on four computers. It seems to work correctly with the latest Mac OS. It has been tested on Linux, but the display is a bit wonky. It has not yet been tested elsewhere.
- I encourage you to explore the script, customize it, and make it your own!
- For Stable Diffusion's Text-to-Image, you must assign a step amount before clicking on 'generate image'.
- For generating animation, fill out all prompts and select a preset. The starting prompt must be at Frame 0.
- Animations take time. To expedite the process, reduce the seed and the total frame count (length).
- Image-to-Animation generates only 48 frames and is rather wonky.

## Support
If you found brAInstormer helpful and would like to support its development, consider buying me a coffee:

[![Support via PayPal](https://www.paypalobjects.com/en_US/i/btn/btn_donateCC_LG.gif)](https://www.paypal.me/noodlebake)


## License
This project is licensed under the MIT License 
