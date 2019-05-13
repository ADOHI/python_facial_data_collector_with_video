# python_facial_data_collector_with_video
DLC MALAB 3rd team
First project

this application will gatter your facial data through webcam. so you must need webcam or something other camera.

## How to start

First, just install required package using pip and requirements.txt
<pre><code>
pip install -r requirements.txt
</code></pre>

And open config.py, revice name and video title that will be shown
(default is sample.mov)

That's all. just run main.py
<pre><code>
python main.py
</code></pre>

It will be saved 3 different images

##### Fisrt. original webcam image

##### Second. cropped facial image

##### Third. cropped, resize(48, 48), grayscale, normalized image for ML,NN
