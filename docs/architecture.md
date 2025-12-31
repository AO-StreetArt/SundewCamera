# Architecture

Sundew prototypes are built on a Raspberry Pi 5 with an AI Hat, containing a Hailo AI processor, although if they are successful we can move onto a SOM from Hailo.ai with minimal work.

As far as software architecture - we need to optimize throughput well in order to keep up with the camera.  We're going to do this at two levels: first, by splitting the output processing from the CV processing, and then within the CV processing we'll be supporting multiple threads operating on different processors.

## Onboard Applications

We're going to have two high-level applications that run on the Raspberry pi - a CV Processor, which performs ML tasks and then broadcasts the results; and an Output Processor, which takes the results and turns them into human-digestable form.  These are both python applications for our prototype, and will communicate via IPC.

## CV Processor

Our CV Processor will be based on the Hailo-apps examples (ie. https://github.com/hailo-ai/hailo-apps/blob/main/hailo_apps/python/standalone_apps/object_detection/object_detection.py).  We'll be following the same core process - a thread dedicated to CV processing on the Hailo processor, and then threads for pre/post processing.  In our case, the post processing thread will be setup to send it's detected objects to a configured IPC address.  To begin with, we will just include object detection.

This will incorporate some basic optimizations - like only processing 1 in every X number of frames.

## Output Processor

The Output Processor accepts IPC messages from the CV Processor, and then turns those into audio messages that can be played to the end user.  Right now, this is non-LLM based processing (so sound files stored on disk and then played to users).  In the future, we'll evaluate if we can leverage a micro LLM on the Hailo board without compromising the performance of the CV features.