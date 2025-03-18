# Python Screen driver module

The purpose of the code is to provide a high speed methodology of capturing screen regions from X11. It does simple damage detection and only feeds blocks of pixels that the user actually needs. 

In the current format the shared module returns encoded jpegs, but is setup with a future mode switch to return alternative image formats and raw or processed bytes like RGB or YUV420. 

# Building

Install deps
```
apt-get update && \
apt-get install -y \
  cmake \
  g++ \
  gcc \
  libjpeg62-turbo-dev \
  libx11-dev \
  libxext-dev \
  libxxhash-dev
  make
```

Run build script: 
```
bash build.sh
```

# Running

This Python example is bare bones along with the client, it expects that you have a way to control the screen you are capturing from, but after building the module you can just run: 

```
python3 screen_cap_example.py
```

The server landing page is http://localhost:9001, with ws listening on 9000. This runs only from localhost as in a real deployment this will be behind a proxy with https. 

The shared object takes configuration parameters IE: 

```
    capture_settings.capture_width = 2560
    capture_settings.capture_height = 1440
    capture_settings.capture_x = 0
    capture_settings.capture_y = 0
    capture_settings.target_fps = 30.0
    capture_settings.jpeg_quality = 40
    capture_settings.paint_over_jpeg_quality = 95
    capture_settings.use_paint_over_quality = True
    capture_settings.paint_over_trigger_frames = 2
    capture_settings.damage_block_threshold = 15
    capture_settings.damage_block_duration = 30
```

Things to know about the jpeg pipeline is that jpeg_quality is the minimum it will use, so under action it gradually reduces to that amount depending on consecutive damaged frames. This will be the same response to backpressure, but will happen more quickly, we see that we are filling up the buffer and on drain we need to step in and reduce framerate and quality to minimum and ramp back up to find the clients capabilities to try and deliver the requested quality and framerate. 

The damage settings control how much we check damage, this can have a profound effect on CPU usage, if you hash everything all the time under 60 fps it eats up a core, but if we ramp down checking on many consecutive damages and ramp back up when the screen is static for long periods of time we still hit a sweet spot where this entire pipeline websockets and all uses a fraction of a core. 
In the above configuration the damage check will only happen every 30 frames if the frame has been static for more than 30, and it will stop checking damage if the frame has been damaged 15 in a row. These all reset instantly on any change from the behavior and go back to normal damage tics. 
