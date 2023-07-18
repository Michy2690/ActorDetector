> Bianco Michael, Bertolini Matteo, Todaro Marco
# Abstract
We present an approach to detect and recognize automatically
the actors present in a movie scene. This method is in
general applicable to all systems where the task is to identify
a person between a list of other subjects, the only need
is to have a proper dataset and fine-tune the model on these
new data. The algorithm also makes use of a generator able
to reconstruct the frontal face if an actor is standing sideways
or if is wearing something that covers his face, for example
a cap or a pair of glasses. Similar task is performed
by Amazon Prime video X-Ray but we focus on the single
frame. This project can help new video services to provide
better user experiences, for instance displaying other useful
information on the actor detected or similar movies in
which he/her acts.

[[Paper]](https://github.com/Michy2690/ActorDetector/blob/main/ACTORS%20DETECTION%20AND%20RETRIEVAL%20IN%20MOVIE%20SCENES/Actor%20recognition%20and%20retrieval%20in%20movie%20scenes.pdf)

<img src="ACTORS DETECTION AND RETRIEVAL IN MOVIE SCENES/images/results.jpg" style="zoom:60%;" />
<img src="ACTORS DETECTION AND RETRIEVAL IN MOVIE SCENES/images/schema.png" style="zoom:60%;" />

# Implementation
~~~bash
pip install -r requirements.txt
~~~
In order to run the code you can download our modified FaceScrub dataset, our pretrained weights and preprocessed data at this [page](https://drive.google.com/drive/folders/1lGaZ4XsTyZgafvOAQYsi7p1Fl1ijbljU).

# Training and fine-tuning
~~~bash
python triplet_parallel.py
~~~
It is necessary to create some directories indicated in this file. 

# Evaluation
~~~bash
python server.py
~~~
Then run the live demo browsing localhost:5000
You can load every movie you want, for higher precision it is recommended to have an HD video in .mp4 format.
