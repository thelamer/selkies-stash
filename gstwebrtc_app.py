# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file incorporates work covered by the following copyright and
# permission notice:
#
#   Copyright 2019 Google LLC
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import asyncio
import base64
import json
import logging
import os
import re
import sys
import time

logger = logging.getLogger("gstwebrtc_app")
logger.setLevel(logging.INFO)

try:
    import gi
    gi.require_version('GLib', "2.0")
    gi.require_version('Gst', "1.0")
    gi.require_version('GstRtp', "1.0")
    gi.require_version('GstSdp', "1.0")
    gi.require_version('GstWebRTC', "1.0")
    from gi.repository import GLib, Gst, GstRtp, GstSdp, GstWebRTC
    fract = Gst.Fraction(60, 1)
    del fract
except Exception as e:
    msg = """ERROR: could not find working GStreamer-Python installation.

If GStreamer is installed at a certain location, set the path to the environment variable GSTREAMER_PATH, then make sure your environment is set correctly using the below commands (for Debian-like distributions):

export GSTREAMER_PATH="${GSTREAMER_PATH:-$(pwd)}"
export PATH="${GSTREAMER_PATH}/bin${PATH:+:${PATH}}"
export LD_LIBRARY_PATH="${GSTREAMER_PATH}/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export GST_PLUGIN_PATH="${GSTREAMER_PATH}/lib/x86_64-linux-gnu/gstreamer-1.0${GST_PLUGIN_PATH:+:${GST_PLUGIN_PATH}}"
export GST_PLUGIN_SYSTEM_PATH="${XDG_DATA_HOME:-${HOME:-~}/.local/share}/gstreamer-1.0/plugins:/usr/lib/x86_64-linux-gnu/gstreamer-1.0${GST_PLUGIN_SYSTEM_PATH:+:${GST_PLUGIN_SYSTEM_PATH}}"
export GI_TYPELIB_PATH="${GSTREAMER_PATH}/lib/x86_64-linux-gnu/girepository-1.0:/usr/lib/x86_64-linux-gnu/girepository-1.0${GI_TYPELIB_PATH:+:${GI_TYPELIB_PATH}}"
export PYTHONPATH="${GSTREAMER_PATH}/lib/python3/dist-packages${PYTHONPATH:+:${PYTHONPATH}}"

Replace "x86_64-linux-gnu" in other architectures manually or use "$(gcc -print-multiarch)" in place.
"""
    logger.error(msg)
    logger.error(e)
    sys.exit(1)
logger.info("GStreamer-Python install looks OK")

class GSTWebRTCAppError(Exception):
    pass

class GSTWebRTCApp:
    def __init__(self, stun_servers=None, turn_servers=None, audio_channels=2, framerate=30, encoder=None, gpu_id=0, video_bitrate=2000, audio_bitrate=96000, keyframe_distance=-1.0, congestion_control=False, video_packetloss_percent=0.0, audio_packetloss_percent=0.0, render_mode="gstreamer"):
        """Initialize GStreamer WebRTC app."""

        self.stun_servers = stun_servers
        self.turn_servers = turn_servers
        self.audio_channels = audio_channels
        self.pipeline = None
        self.webrtcbin = None
        self.data_channel = None
        self.rtpgccbwe = None
        self.congestion_control = congestion_control
        self.encoder = encoder
        self.gpu_id = gpu_id
        self.render_mode = render_mode

        self.framerate = framerate
        self.video_bitrate = video_bitrate
        self.audio_bitrate = audio_bitrate

        self.keyframe_distance = keyframe_distance
        self.min_keyframe_frame_distance = 60
        self.keyframe_frame_distance = -1 if self.keyframe_distance == -1.0 else max(self.min_keyframe_frame_distance, int(self.framerate * self.keyframe_distance))
        self.vbv_multiplier_nv = 1.5 if self.keyframe_distance == -1.0 else 3
        self.vbv_multiplier_va = 1.5 if self.keyframe_distance == -1.0 else 3
        self.vbv_multiplier_vp = 1.5 if self.keyframe_distance == -1.0 else 3
        self.vbv_multiplier_sw = 1.5 if self.keyframe_distance == -1.0 else 3
        self.video_packetloss_percent = video_packetloss_percent
        self.audio_packetloss_percent = audio_packetloss_percent
        self.fec_video_bitrate = int(self.video_bitrate / (1.0 + (self.video_packetloss_percent / 100.0)))
        self.fec_audio_bitrate = int(self.audio_bitrate * (1.0 + (self.audio_packetloss_percent / 100.0)))

        self.on_ice = lambda mlineindex, candidate: logger.warn('unhandled ice event')
        self.on_sdp = lambda sdp_type, sdp: logger.warn('unhandled sdp event')

        self.on_data_open = lambda: logger.warn('unhandled on_data_open')
        self.on_data_close = lambda: logger.warn('unhandled on_data_close')
        self.on_data_error = lambda: logger.warn('unhandled on_data_error')
        self.on_data_message = lambda msg: logger.warn('unhandled on_data_message')

        Gst.init(None)

        self.check_plugins()

        self.ximagesrc = None
        self.ximagesrc_caps = None
        self.last_cursor_sent = None
        self.jpeg_send_task = None
        self.capture_module_instance = None
        self.is_jpeg_capturing = False
        self.current_jpeg_queue = None

    def stop_ximagesrc(self):
        if self.ximagesrc:
            self.ximagesrc.set_state(Gst.State.NULL)

    def start_ximagesrc(self):
        if self.ximagesrc:
            self.ximagesrc.set_property("endx", 0)
            self.ximagesrc.set_property("endy", 0)
            self.ximagesrc.set_state(Gst.State.PLAYING)

    def build_webrtcbin_pipeline(self, audio_only=False):
        self.webrtcbin = Gst.ElementFactory.make("webrtcbin", "app")
        self.webrtcbin.set_property("bundle-policy", "max-compat")
        self.webrtcbin.set_property("latency", 0)

        if self.congestion_control and not audio_only and self.render_mode == "gstreamer":
            self.webrtcbin.connect(
                'request-aux-sender', lambda webrtcbin, dtls_transport: self.__request_aux_sender_gcc(webrtcbin, dtls_transport))
        self.webrtcbin.connect(
            'on-negotiation-needed', lambda webrtcbin: self.__on_negotiation_needed(webrtcbin))
        self.webrtcbin.connect('on-ice-candidate', lambda webrtcbin, mlineindex,
                               candidate: self.__send_ice(webrtcbin, mlineindex, candidate))

        if self.stun_servers:
            logger.info("updating STUN server")
            self.webrtcbin.set_property("stun-server", self.stun_servers[0])
        if self.turn_servers:
            for i, turn_server in enumerate(self.turn_servers):
                logger.info("updating TURN server")
                if i == 0:
                    self.webrtcbin.set_property("turn-server", self.turn_server)
                else:
                    self.webrtcbin.emit("add-turn-server", turn_server)

        self.pipeline.add(self.webrtcbin)

    def build_video_pipeline(self):
        if self.render_mode == "jpeg-datachannel":
            logger.info("Skipping video pipeline build for jpeg-datachannel mode.")
            return

        self.ximagesrc = Gst.ElementFactory.make("ximagesrc", "x11")
        ximagesrc = self.ximagesrc
        ximagesrc.set_property("show-pointer", 0)
        ximagesrc.set_property("remote", 1)
        ximagesrc.set_property("blocksize", 16384)
        ximagesrc.set_property("use-damage", 0)
        self.ximagesrc_caps = Gst.caps_from_string("video/x-raw")
        self.ximagesrc_caps.set_value("framerate", Gst.Fraction(self.framerate, 1))
        self.ximagesrc_capsfilter = Gst.ElementFactory.make("capsfilter")
        self.ximagesrc_capsfilter.set_property("caps", self.ximagesrc_caps)

        if self.encoder in ["nvh264enc"]:
            cudaupload = Gst.ElementFactory.make("cudaupload")
            if self.gpu_id >= 0:
                cudaupload.set_property("cuda-device-id", self.gpu_id)
            cudaconvert = Gst.ElementFactory.make("cudaconvert")
            if self.gpu_id >= 0:
                cudaconvert.set_property("cuda-device-id", self.gpu_id)
            cudaconvert.set_property("qos", True)
            cudaconvert_caps = Gst.caps_from_string("video/x-raw(memory:CUDAMemory)")
            cudaconvert_capsfilter = Gst.ElementFactory.make("capsfilter")
            cudaconvert_capsfilter.set_property("caps", cudaconvert_caps)
            nvh264enc = Gst.ElementFactory.make("nvh264enc", "nvenc")
            nvh264enc.set_property("bitrate", self.fec_video_bitrate)
            nvh264enc.set_property("rc-mode", "cbr")
            nvh264enc.set_property("gop-size", -1 if self.keyframe_distance == -1.0 else self.keyframe_frame_distance)
            nvh264enc.set_property("strict-gop", True)
            nvh264enc.set_property("aud", False)
            nvh264enc.set_property("b-adapt", False)
            nvh264enc.set_property("rc-lookahead", 0)
            nvh264enc.set_property("vbv-buffer-size", int((self.fec_video_bitrate + self.framerate - 1) // self.framerate * self.vbv_multiplier_nv))
            nvh264enc.set_property("bframes", 0)
            nvh264enc.set_property("zerolatency", True)
            nvh264enc.set_property("cabac", True)
            nvh264enc.set_property("repeat-sequence-header", True)
            nvh264enc.set_property("preset", "p4")
            nvh264enc.set_property("tune", "ultra-low-latency")
            nvh264enc.set_property("multi-pass", "two-pass-quarter")
            h264enc_caps = Gst.caps_from_string("video/x-h264")
            h264enc_caps.set_value("profile", "main")
            h264enc_caps.set_value("stream-format", "byte-stream")
            h264enc_capsfilter = Gst.ElementFactory.make("capsfilter")
            h264enc_capsfilter.set_property("caps", h264enc_caps)
            rtph264pay = Gst.ElementFactory.make("rtph264pay")
            rtph264pay.set_property("mtu", 1200)
            rtph264pay.set_property("aggregate-mode", "zero-latency")
            rtph264pay.set_property("config-interval", -1)
            extensions_return = self.rtp_add_extensions(rtph264pay)
            if not extensions_return:
                logger.warning("WebRTC RTP extension configuration failed with video, this may lead to suboptimal performance")
            rtph264pay_caps = Gst.caps_from_string("application/x-rtp")
            rtph264pay_capsfilter = Gst.ElementFactory.make("capsfilter")
            rtph264pay_capsfilter.set_property("caps", rtph264pay_caps)
            pipeline_elements = [ximagesrc, ximagesrc_capsfilter, cudaupload, cudaconvert, cudaconvert_capsfilter, nvh264enc, h264enc_capsfilter, rtph264pay, rtph264pay_capsfilter]


        elif self.encoder in ["x264enc"]:
            videoconvert = Gst.ElementFactory.make("videoconvert")
            videoconvert_capsfilter = Gst.ElementFactory.make("capsfilter")
            videoconvert_capsfilter.set_property("caps", Gst.caps_from_string("video/x-raw,format=NV12"))
            x264enc = Gst.ElementFactory.make("x264enc", "x264enc")
            x264enc.set_property("threads", min(4, max(1, len(os.sched_getaffinity(0)) - 1)))
            x264enc.set_property("aud", False)
            x264enc.set_property("b-adapt", False)
            x264enc.set_property("bframes", 0)
            x264enc.set_property("dct8x8", False)
            x264enc.set_property("insert-vui", True)
            x264enc.set_property("key-int-max", 2147483647 if self.keyframe_distance == -1.0 else self.keyframe_frame_distance)
            x264enc.set_property("mb-tree", False)
            x264enc.set_property("rc-lookahead", 0)
            x264enc.set_property("sync-lookahead", 0)
            x264enc.set_property("vbv-buf-capacity", int((1000 + self.framerate - 1) // self.framerate * self.vbv_multiplier_sw))
            x264enc.set_property("sliced-threads", True)
            x264enc.set_property("byte-stream", True)
            x264enc.set_property("pass", "cbr")
            x264enc.set_property("speed-preset", "ultrafast")
            x264enc.set_property("tune", "zerolatency")
            x264enc.set_property("bitrate", self.fec_video_bitrate)
            h264enc_capsfilter = Gst.ElementFactory.make("capsfilter")
            h264enc_capsfilter.set_property("caps", Gst.caps_from_string("video/x-h264,profile=main,stream-format=byte-stream"))
            rtph264pay = Gst.ElementFactory.make("rtph264pay")
            rtph264pay.set_property("mtu", 1200)
            rtph264pay.set_property("aggregate-mode", "zero-latency")
            rtph264pay.set_property("config-interval", -1)
            extensions_return = self.rtp_add_extensions(rtph264pay)
            if not extensions_return:
                logger.warning("WebRTC RTP extension configuration failed with video, this may lead to suboptimal performance")
            rtph264pay_capsfilter = Gst.ElementFactory.make("capsfilter")
            rtph264pay_capsfilter.set_property("caps", Gst.caps_from_string("application/x-rtp,media=video,clock-rate=90000,encoding-name=H264,payload=97,rtcp-fb-nack-pli=true,rtcp-fb-ccm-fir=true,rtcp-fb-x-gstreamer-fir-as-repair=true"))
            pipeline_elements = [ximagesrc, ximagesrc_capsfilter, videoconvert, videoconvert_capsfilter , x264enc, h264enc_capsfilter, rtph264pay, rtph264pay_capsfilter]

        else:
            raise GSTWebRTCAppError("Unsupported encoder for pipeline: %s" % self.encoder)

        for pipeline_element in pipeline_elements:
            self.pipeline.add(pipeline_element)

        pipeline_elements += [self.webrtcbin]
        for i in range(len(pipeline_elements) - 1):
            if not Gst.Element.link(pipeline_elements[i], pipeline_elements[i + 1]):
                raise GSTWebRTCAppError("Failed to link {} -> {}".format(pipeline_elements[i].get_name(), pipeline_elements[i + 1].get_name()))

        transceiver = self.webrtcbin.emit("get-transceiver", 0)
        transceiver.set_property("do-nack", True)
        transceiver.set_property("fec-type", GstWebRTC.WebRTCFECType.ULP_RED if self.video_packetloss_percent > 0 else GstWebRTC.WebRTCFECType.NONE)
        transceiver.set_property("fec-percentage", self.video_packetloss_percent)


    async def send_jpeg_datachannel(self, jpeg_bytes):
        """Sends JPEG bytes over the datachannel as ArrayBuffer."""
        if not self.is_data_channel_ready():
            logger.debug("Data channel not ready, dropping JPEG frame.")
            return
        if self.data_channel:
            self.data_channel.emit("send-data", GLib.Bytes.new(jpeg_bytes))

    async def _jpeg_stream_task(self, jpeg_queue):
        """Background task to continuously send JPEGs from the queue."""
        while self.is_jpeg_capturing:
            try:
                jpeg_bytes = await jpeg_queue.get()
                if jpeg_bytes:
                    await self.send_jpeg_datachannel(jpeg_bytes)
                jpeg_queue.task_done()
            except asyncio.CancelledError:
                logger.info("JPEG send task cancelled.")
                break
            except Exception as e:
                logger.error(f"Error sending JPEG frame: {e}")
                break

    def start_jpeg_datachannel_stream(self, capture_module, capture_settings, stripe_callback, jpeg_queue):
        """Starts the JPEG datachannel streaming mode."""
        logger.info("Starting JPEG datachannel stream.")
        self.capture_module_instance = capture_module
        self.current_jpeg_queue = jpeg_queue
        global loop
        self.is_jpeg_capturing = True
        capture_module.start_capture(capture_settings, stripe_callback)
        self.jpeg_send_task = asyncio.create_task(self._jpeg_stream_task(jpeg_queue))


    def build_audio_pipeline(self):
        pulsesrc = Gst.ElementFactory.make("pulsesrc", "pulsesrc")
        pulsesrc.set_property("provide-clock", True)
        pulsesrc.set_property("do-timestamp", False)
        pulsesrc.set_property("buffer-time", 100000)
        pulsesrc.set_property("latency-time", 1000)
        pulsesrc_capsfilter = Gst.ElementFactory.make("capsfilter")
        pulsesrc_capsfilter.set_property("caps", Gst.caps_from_string("audio/x-raw,channels=%d" % self.audio_channels))
        opusenc = Gst.ElementFactory.make("opusenc", "opusenc")
        opusenc.set_property("audio-type", "restricted-lowdelay")
        opusenc.set_property("bandwidth", "fullband")
        opusenc.set_property("bitrate-type", "cbr")
        opusenc.set_property("frame-size", "10")
        opusenc.set_property("perfect-timestamp", True)
        opusenc.set_property("max-payload-size", 4000)
        opusenc.set_property("inband-fec", self.audio_packetloss_percent > 0)
        opusenc.set_property("packet-loss-percentage", self.audio_packetloss_percent)
        opusenc.set_property("bitrate", self.audio_bitrate)
        rtpopuspay = Gst.ElementFactory.make("rtpopuspay")
        rtpopuspay.set_property("mtu", 1200)
        extensions_return = self.rtp_add_extensions(rtpopuspay, audio=True)
        if not extensions_return:
            logger.warning("WebRTC RTP extension configuration failed with audio, this may lead to suboptimal performance")
        rtpopuspay_queue = Gst.ElementFactory.make("queue", "rtpopuspay_queue")
        rtpopuspay_queue.set_property("leaky", "downstream")
        rtpopuspay_queue.set_property("flush-on-eos", True)
        rtpopuspay_queue.set_property("max-size-time", 16000000)
        rtpopuspay_queue.set_property("max-size-buffers", 0)
        rtpopuspay_queue.set_property("max-size-bytes", 0)
        rtpopuspay_capsfilter = Gst.ElementFactory.make("capsfilter")
        rtpopuspay_capsfilter.set_property("caps", Gst.caps_from_string("application/x-rtp,media=audio,encoding-name=OPUS,payload=111,clock-rate=48000"))
        pipeline_elements = [pulsesrc, pulsesrc_capsfilter, opusenc, rtpopuspay, rtpopuspay_queue, rtpopuspay_capsfilter]

        for pipeline_element in pipeline_elements:
            self.pipeline.add(pipeline_element)

        pipeline_elements += [self.webrtcbin]
        for i in range(len(pipeline_elements) - 1):
            if not Gst.Element.link(pipeline_elements[i], pipeline_elements[i + 1]):
                raise GSTWebRTCAppError("Failed to link {} -> {}".format(pipeline_elements[i].get_name(), pipeline_elements[i + 1].get_name()))


    def check_plugins(self):
        required = ["opus", "nice", "webrtc", "app", "dtls", "srtp", "rtp", "sctp", "rtpmanager"]
        supported = ["nvh264enc", "nvh265enc", "nvav1enc", "vah264enc", "vah265enc", "vavp9enc", "vaav1enc", "x264enc", "openh264enc", "x265enc", "vp8enc", "vp9enc", "svtav1enc", "av1enc", "rav1enc"]
        if self.encoder not in supported and self.render_mode == "gstreamer":
            raise GSTWebRTCAppError('Unsupported encoder, must be one of: ' + ','.join(supported))

        if ("av1" in self.encoder or self.congestion_control) and self.render_mode == "gstreamer":
            required.append("rsrtp")

        if self.encoder.startswith("nv") and self.render_mode == "gstreamer":
            required.append("nvcodec")
        elif self.encoder.startswith("va") and self.render_mode == "gstreamer":
            required.append("va")
        elif self.encoder in ["x264enc"] and self.render_mode == "gstreamer":
            required.append("x264")
        elif self.encoder in ["openh264enc"] and self.render_mode == "gstreamer":
            required.append("openh264")
        elif self.encoder in ["x265enc"] and self.render_mode == "gstreamer":
            required.append("x265")
        elif self.encoder in ["vp8enc", "vp9enc"] and self.render_mode == "gstreamer":
            required.append("vpx")
        elif self.encoder in ["svtav1enc"] and self.render_mode == "gstreamer":
            required.append("svtav1")
        elif self.encoder in ["av1enc"] and self.render_mode == "gstreamer":
            required.append("aom")
        elif self.encoder in ["rav1enc"] and self.render_mode == "gstreamer":
            required.append("rav1e")


        missing = list(
            filter(lambda p: Gst.Registry.get().find_plugin(p) is None, required))
        if missing:
            raise GSTWebRTCAppError('Missing gstreamer plugins:', missing)

    def set_sdp(self, sdp_type, sdp):
        if not self.webrtcbin:
            raise GSTWebRTCAppError('Received SDP before session started')
        if sdp_type != 'answer':
            raise GSTWebRTCAppError('ERROR: sdp type was not "answer"')

        _, sdpmsg = GstSdp.SDPMessage.new_from_text(sdp)
        answer = GstWebRTC.WebRTCSessionDescription.new(
            GstWebRTC.WebRTCSDPType.ANSWER, sdpmsg)
        promise = Gst.Promise.new()
        self.webrtcbin.emit('set-remote-description', answer, promise)
        promise.interrupt()

    def set_ice(self, mlineindex, candidate):
        logger.info("setting ICE candidate: %d, %s" % (mlineindex, candidate))
        if not self.webrtcbin:
            raise GSTWebRTCAppError('Received ICE before session started')
        self.webrtcbin.emit('add-ice-candidate', mlineindex, candidate)

    def set_framerate(self, framerate):
        if self.pipeline:
            self.framerate = framerate
            self.keyframe_frame_distance = -1 if self.keyframe_distance == -1.0 else max(self.min_keyframe_frame_distance, int(self.framerate * self.keyframe_distance))
            if self.encoder.startswith("nv"):
                element = Gst.Bin.get_by_name(self.pipeline, "nvenc")
                element.set_property("gop-size", -1 if self.keyframe_distance == -1.0 else self.keyframe_frame_distance)
                element.set_property("vbv-buffer-size", int((self.fec_video_bitrate + self.framerate - 1) // self.framerate * self.vbv_multiplier_nv))
            elif self.encoder.startswith("va"):
                element = Gst.Bin.get_by_name(self.pipeline, "vaenc")
                element.set_property("key-int-max", 1024 if self.keyframe_distance == -1.0 else self.keyframe_frame_distance)
                element.set_property("cpb-size", int((self.fec_video_bitrate + self.framerate - 1) // self.framerate * self.vbv_multiplier_va))
            elif self.encoder in ["x264enc"]:
                element = Gst.Bin.get_by_name(self.pipeline, "x264enc")
                element.set_property("key-int-max", 2147483647 if self.keyframe_distance == -1.0 else self.keyframe_frame_distance)
                element.set_property("vbv-buf-capacity", int((1000 + self.framerate - 1) // self.framerate * self.vbv_multiplier_sw))
            elif self.encoder in ["openh264enc"]:
                element = Gst.Bin.get_by_name(self.pipeline, "openh264enc")
                element.set_property("gop-size", 2147483647 if self.keyframe_distance == -1.0 else self.keyframe_frame_distance)
            elif self.encoder in ["x265enc"]:
                element = Gst.Bin.get_by_name(self.pipeline, "x265enc")
                element.set_property("key-int-max", 2147483647 if self.keyframe_distance == -1.0 else self.keyframe_frame_distance)
            elif self.encoder.startswith("vp"):
                element = Gst.Bin.get_by_name(self.pipeline, "vpenc")
                element.set_property("keyframe-max-dist", 2147483647 if self.keyframe_distance == -1.0 else self.keyframe_frame_distance)
                vbv_buffer_size = int((1000 + self.framerate - 1) // self.framerate * self.vbv_multiplier_vp)
                element.set_property("buffer-initial-size", vbv_buffer_size)
                element.set_property("buffer-optimal-size", vbv_buffer_size)
                element.set_property("buffer-size", vbv_buffer_size)
            elif self.encoder in ["svtav1enc"]:
                element = Gst.Bin.get_by_name(self.pipeline, "svtav1enc")
                element.set_property("intra-period-length", -1 if self.keyframe_distance == -1.0 else self.keyframe_frame_distance)
            elif self.encoder in ["av1enc"]:
                element = Gst.Bin.get_by_name(self.pipeline, "av1enc")
                element.set_property("keyframe-max-dist", 2147483647 if self.keyframe_distance == -1.0 else self.keyframe_frame_distance)
            elif self.encoder in ["rav1enc"]:
                element = Gst.Bin.get_by_name(self.pipeline, "rav1enc")
                element.set_property("max-key-frame-interval", 715827882 if self.keyframe_distance == -1.0 else self.keyframe_frame_distance)
            else:
                logger.warning("setting keyframe interval (GOP size) not supported with encoder: %s" % self.encoder)

            self.ximagesrc_caps = Gst.caps_from_string("video/x-raw")
            self.ximagesrc_caps.set_value("framerate", Gst.Fraction(self.framerate, 1))
            self.ximagesrc_capsfilter.set_property("caps", self.ximagesrc_caps)
            logger.info("framerate set to: %d" % framerate)

    def set_video_bitrate(self, bitrate, cc=False):
        if self.pipeline:
            fec_bitrate = int(bitrate / (1.0 + (self.video_packetloss_percent / 100.0)))
            if (not cc) and self.congestion_control and self.rtpgccbwe is not None:
                self.rtpgccbwe.set_property("min-bitrate", max(100000 + self.fec_audio_bitrate, int(bitrate * 1000 * 0.1 + self.fec_audio_bitrate)))
                self.rtpgccbwe.set_property("max-bitrate", int(bitrate * 1000 + self.fec_audio_bitrate))
                self.rtpgccbwe.set_property("estimated-bitrate", int(bitrate * 1000 + self.fec_audio_bitrate))
            if self.encoder.startswith("nv"):
                element = Gst.Bin.get_by_name(self.pipeline, "nvenc")
                if not cc:
                    element.set_property("vbv-buffer-size", int((fec_bitrate + self.framerate - 1) // self.framerate * self.vbv_multiplier_nv))
                element.set_property("bitrate", fec_bitrate)
            elif self.encoder.startswith("va"):
                element = Gst.Bin.get_by_name(self.pipeline, "vaenc")
                if not cc:
                    element.set_property("cpb-size", int((fec_bitrate + self.framerate - 1) // self.framerate * self.vbv_multiplier_va))
                element.set_property("bitrate", fec_bitrate)
            elif self.encoder in ["x264enc"]:
                element = Gst.Bin.get_by_name(self.pipeline, "x264enc")
                element.set_property("bitrate", fec_bitrate)
            elif self.encoder in ["openh264enc"]:
                element = Gst.Bin.get_by_name(self.pipeline, "openh264enc")
                element.set_property("bitrate", fec_bitrate * 1000)
            elif self.encoder in ["x265enc"]:
                element = Gst.Bin.get_by_name(self.pipeline, "x265enc")
                element.set_property("bitrate", fec_bitrate)
            elif self.encoder in ["vp8enc", "vp9enc"]:
                element = Gst.Bin.get_by_name(self.pipeline, "vpenc")
                element.set_property("target-bitrate", fec_bitrate * 1000)
            elif self.encoder in ["svtav1enc"]:
                element = Gst.Bin.get_by_name(self.pipeline, "svtav1enc")
                element.set_property("target-bitrate", fec_bitrate)
            elif self.encoder in ["av1enc"]:
                element = Gst.Bin.get_by_name(self.pipeline, "av1enc")
                element.set_property("target-bitrate", fec_bitrate)
            elif self.encoder in ["rav1enc"]:
                element = Gst.Bin.get_by_name(self.pipeline, "rav1enc")
                element.set_property("bitrate", fec_bitrate * 1000)
            else:
                logger.warning("set_video_bitrate not supported with encoder: %s" % self.encoder)

            if not cc:
                logger.info("video bitrate set to: %d" % bitrate)
            else:
                logger.debug("video bitrate set with congestion control to: %d" % bitrate)

            self.video_bitrate = bitrate
            self.fec_video_bitrate = fec_bitrate

            if not cc:
                self.__send_data_channel_message(
                    "pipeline", {"status": "Video bitrate set to: %d" % bitrate})


    def set_audio_bitrate(self, bitrate):
        if self.pipeline:
            fec_bitrate = int(bitrate * (1.0 + (self.audio_packetloss_percent / 100.0)))
            if self.congestion_control and self.rtpgccbwe is not None:
                self.rtpgccbwe.set_property("min-bitrate", max(100000 + fec_bitrate, int(self.video_bitrate * 1000 * 0.1 + fec_bitrate)))
                self.rtpgccbwe.set_property("max-bitrate", int(self.video_bitrate * 1000 + fec_bitrate))
                self.rtpgccbwe.set_property("estimated-bitrate", int(self.video_bitrate * 1000 + fec_bitrate))
            element = Gst.Bin.get_by_name(self.pipeline, "opusenc")
            element.set_property("bitrate", bitrate)

            logger.info("audio bitrate set to: %d" % bitrate)
            self.audio_bitrate = bitrate
            self.fec_audio_bitrate = fec_bitrate

            self.__send_data_channel_message(
                "pipeline", {"status": "Audio bitrate set to: %d" % bitrate})

    def set_pointer_visible(self, visible):
        element = Gst.Bin.get_by_name(self.pipeline, "x11")
        element.set_property("show-pointer", visible)
        self.__send_data_channel_message(
            "pipeline", {"status": "Set pointer visibility to: %d" % visible})

    def send_clipboard_data(self, data):
        CLIPBOARD_RESTRICTION = 65400
        clipboard_message = base64.b64encode(data.encode()).decode("utf-8")
        clipboard_length = len(clipboard_message)
        if clipboard_length <= CLIPBOARD_RESTRICTION:
            self.__send_data_channel_message(
                "clipboard", {"content": clipboard_message})
        else:
            logger.warning("clipboard may not be sent to the client because the base64 message length {} is above the maximum length of {}".format(clipboard_length, CLIPBOARD_RESTRICTION))

    def send_cursor_data(self, data):
        self.last_cursor_sent = data
        self.__send_data_channel_message(
            "cursor", data)

    def send_gpu_stats(self, load, memory_total, memory_used):
        self.__send_data_channel_message("gpu_stats", {
            "load": load,
            "memory_total": memory_total,
            "memory_used": memory_used,
        })
    def send_reload_window(self):
        logger.info("sending window reload")
        self.__send_data_channel_message(
            "system", {"action": "reload"})
    def send_framerate(self, framerate):
        logger.info("sending framerate")
        self.__send_data_channel_message(
            "system", {"action": "framerate,"+str(framerate)})
    def send_video_bitrate(self, bitrate):
        logger.info("sending video bitrate")
        self.__send_data_channel_message(
            "system", {"action": "video_bitrate,%d" % bitrate})
    def send_audio_bitrate(self, bitrate):
        logger.info("sending audio bitrate")
        self.__send_data_channel_message(
            "system", {"action": "audio_bitrate,%d" % bitrate})
    def send_encoder(self, encoder):
        logger.info("sending encoder: " + encoder)
        self.__send_data_channel_message(
            "system", {"action": "encoder,%s" % encoder})
    def send_resize_enabled(self, resize_enabled):
        logger.info("sending resize enabled state")
        self.__send_data_channel_message(
            "system", {"action": "resize,"+str(resize_enabled)})
    def send_remote_resolution(self, res):
        logger.info("sending remote resolution of: " + res)
        self.__send_data_channel_message(
            "system", {"action": "resolution," + res})
    def send_ping(self, t):
        self.__send_data_channel_message(
            "ping", {"start_time": float("%.3f" % t)})
    def send_latency_time(self, latency):
        self.__send_data_channel_message(
            "latency_measurement", {"latency_ms": latency})
    def send_system_stats(self, cpu_percent, mem_total, mem_used):
        self.__send_data_channel_message(
            "system_stats", {
                "cpu_percent": cpu_percent,
                "mem_total": mem_total,
                "mem_used": mem_used,
            })

    def is_data_channel_ready(self):
        return self.data_channel and self.data_channel.get_property("ready-state") == GstWebRTC.WebRTCDataChannelState.OPEN

    def __send_data_channel_message(self, msg_type, data):
        if not self.is_data_channel_ready():
            logger.debug(
                "skipping message because data channel is not ready: %s" % msg_type)
            return

        msg = {"type": msg_type, "data": data}
        self.data_channel.emit("send-string", json.dumps(msg))

    def __on_offer_created(self, promise, _, __):
        promise.wait()
        reply = promise.get_reply()
        offer = reply.get_value('offer')
        promise = Gst.Promise.new()
        self.webrtcbin.emit('set-local-description', offer, promise)
        promise.interrupt()
        loop = asyncio.new_event_loop()
        sdp_text = offer.sdp.as_text()
        if 'rtx-time' not in sdp_text:
            logger.warning("injecting rtx-time to SDP")
            sdp_text = re.sub(r'(apt=\d+)', r'\1;rtx-time=125', sdp_text)
        elif 'rtx-time=125' not in sdp_text:
            logger.warning("injecting modified rtx-time to SDP")
            sdp_text = re.sub(r'rtx-time=\d+', r'rtx-time=125', sdp_text)
        if "h264" in self.encoder or "x264" in self.encoder:
            if 'profile-level-id' not in sdp_text:
                logger.warning("injecting profile-level-id to SDP")
                sdp_text = sdp_text.replace('packetization-mode=', 'profile-level-id=42e01f;packetization-mode=')
            elif 'profile-level-id=42e01f' not in sdp_text:
                logger.warning("injecting modified profile-level-id to SDP")
                sdp_text = re.sub(r'profile-level-id=\w+', r'profile-level-id=42e01f', sdp_text)
            if 'level-asymmetry-allowed' not in sdp_text:
                logger.warning("injecting level-asymmetry-allowed to SDP")
                sdp_text = sdp_text.replace('packetization-mode=', 'level-asymmetry-allowed=1;packetization-mode=')
            elif 'level-asymmetry-allowed=1' not in sdp_text:
                logger.warning("injecting modified level-asymmetry-allowed to SDP")
                sdp_text = re.sub(r'level-asymmetry-allowed=\d+', r'level-asymmetry-allowed=1', sdp_text)
        if "h264" in self.encoder or "x264" in self.encoder or "h265" in self.encoder or "x265" in self.encoder:
            if 'sps-pps-idr-in-keyframe' not in sdp_text:
                logger.warning("injecting sps-pps-idr-in-keyframe to SDP")
                sdp_text = sdp_text.replace('packetization-mode=', 'sps-pps-idr-in-keyframe=1;packetization-mode=')
            elif 'sps-pps-idr-in-keyframe=1' not in sdp_text:
                logger.warning("injecting modified sps-pps-idr-in-keyframe to SDP")
                sdp_text = re.sub(r'sps-pps-idr-in-keyframe=\d+', r'sps-pps-idr-in-keyframe=1', sdp_text)
        if "opus/" in sdp_text.lower():
            sdp_text = re.sub(r'([^-]sprop-[^\r\n]+)', r'\1\r\na=ptime:10', sdp_text)
        loop.run_until_complete(self.on_sdp('offer', sdp_text))

    def __request_aux_sender_gcc(self, webrtcbin, dtls_transport):
        self.rtpgccbwe = Gst.ElementFactory.make("rtpgccbwe")
        if self.rtpgccbwe is None:
            logger.warning("rtpgccbwe element is not available, not performing any congestion control.")
            return None
        logger.info("handling on-request-aux-header, activating rtpgccbwe congestion control.")
        self.rtpgccbwe.set_property("min-bitrate", max(100000 + self.fec_audio_bitrate, int(self.video_bitrate * 1000 * 0.1 + self.fec_audio_bitrate)))
        self.rtpgccbwe.set_property("max-bitrate", int(self.video_bitrate * 1000 + self.fec_audio_bitrate))
        self.rtpgccbwe.set_property("estimated-bitrate", int(self.video_bitrate * 1000 + self.fec_audio_bitrate))
        self.rtpgccbwe.connect("notify::estimated-bitrate", lambda bwe, pspec: self.set_video_bitrate(int((bwe.get_property(pspec.name) - self.fec_audio_bitrate) / 1000), cc=True))
        return self.rtpgccbwe

    def rtp_add_extensions(self, payloader, audio=False):
        rtp_id_iteration = 0
        return_result = True
        custom_ext = {"http://www.webrtc.org/experiments/rtp-hdrext/playout-delay": self.PlayoutDelayExtension()}

        rtp_uri_list = []
        if self.congestion_control:
            rtp_uri_list += ["http://www.ietf.org/id/draft-holmer-rmcat-transport-wide-cc-extensions-01"]
        if not audio:
            rtp_uri_list += ["http://www.webrtc.org/experiments/rtp-hdrext/playout-delay"]
        for rtp_uri in rtp_uri_list:
            try:
                rtp_id = self.__pick_rtp_extension_id(payloader, rtp_uri, previous_rtp_id=rtp_id_iteration)
                if rtp_id is not None:
                    if rtp_uri in custom_ext.keys():
                        rtp_extension = custom_ext[rtp_uri]
                    else:
                        rtp_extension = GstRtp.RTPHeaderExtension.create_from_uri(rtp_uri)
                    if not rtp_extension:
                        raise GSTWebRTCAppError("GstRtp.RTPHeaderExtension for {} is None".format(rtp_uri))
                    rtp_extension.set_id(rtp_id)
                    payloader.emit("add-extension", rtp_extension)
                    rtp_id_iteration = rtp_id
            except Exception as e:
                logger.warning("RTP extension {} not added because of error {}".format(rtp_uri, e))
                return_result = False
        return return_result

    def __pick_rtp_extension_id(self, payloader, uri, previous_rtp_id=0):
        payloader_properties = payloader.list_properties()
        enabled_extensions = payloader.get_property("extensions") if "extensions" in [payloader_property.name for payloader_property in payloader_properties] else None
        if not enabled_extensions:
            logger.debug("'extensions' property in {} does not exist in payloader, application code must ensure to select non-conflicting IDs for any additionally configured extensions".format(payloader.get_name()))
            return max(1, previous_rtp_id + 1)
        extension = next((ext for ext in enabled_extensions if ext.get_uri() == uri), None)
        if extension:
            return None
        used_numbers = set(ext.get_id() for ext in enabled_extensions)
        num = 1
        while True:
            if num not in used_numbers:
                return num
            num += 1

    def __on_negotiation_needed(self, webrtcbin):
        logger.info("handling on-negotiation-needed, creating offer.")
        promise = Gst.Promise.new_with_change_func(
            self.__on_offer_created, webrtcbin, None)
        webrtcbin.emit('create-offer', None, promise)

    def __send_ice(self, webrtcbin, mlineindex, candidate):
        logger.debug("received ICE candidate: %d %s", mlineindex, candidate)
        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.on_ice(mlineindex, candidate))

    def bus_call(self, message):
        t = message.type
        if t == Gst.MessageType.EOS:
            logger.error("End-of-stream\n")
            return False
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error("Error: %s: %s\n" % (err, debug))
            return False
        elif t == Gst.MessageType.STATE_CHANGED:
            if isinstance(message.src, Gst.Pipeline):
                old_state, new_state, pending_state = message.parse_state_changed()
                logger.info(("Pipeline state changed from %s to %s." %
                    (old_state.value_nick, new_state.value_nick)))
                if (old_state.value_nick == "paused" and new_state.value_nick == "ready"):
                    logger.info("stopping bus message loop")
                    return False
        elif t == Gst.MessageType.LATENCY:
            if self.webrtcbin:
                self.webrtcbin.set_property("latency", 0)
        return True

    def start_pipeline(self, audio_only=False):
        logger.info("starting pipeline")

        self.pipeline = Gst.Pipeline.new()

        self.build_webrtcbin_pipeline(audio_only)

        if audio_only:
            self.build_audio_pipeline()
        elif self.render_mode == "gstreamer":
            self.build_video_pipeline()
        elif self.render_mode == "jpeg-datachannel":
            logger.info("JPEG Datachannel Mode: Video pipeline will not be built by GStreamer.")


        res = self.pipeline.set_state(Gst.State.PLAYING)
        if res != Gst.StateChangeReturn.SUCCESS:
            raise GSTWebRTCAppError(
                "Failed to transition pipeline to PLAYING: %s" % res)

        if not audio_only:
            options = Gst.Structure("application/data-channel")
            options.set_value("ordered", True)
            options.set_value("priority", "high")
            options.set_value("max-retransmits", 0)
            self.data_channel = self.webrtcbin.emit(
                'create-data-channel', "input", options)
            self.data_channel.connect('on-open', lambda _: self.on_data_open())
            self.data_channel.connect('on-close', lambda _: self.on_data_close())
            self.data_channel.connect('on-error', lambda _: self.on_data_error())
            self.data_channel.connect(
                'on-message-string', lambda _, msg: self.on_data_message(msg))

        logger.info("{} pipeline started".format("audio" if audio_only else "video"))

    async def handle_bus_calls(self):
        running = True
        bus = None
        while running:
            if self.pipeline is not None:
                bus = self.pipeline.get_bus()
            if bus is not None:
                while bus.have_pending():
                    msg = bus.pop()
                    if not self.bus_call(msg):
                        running = False
            await asyncio.sleep(0.1)

    def stop_pipeline(self):
        logger.info("stopping pipeline")
        if self.render_mode == "jpeg-datachannel":
            logger.info("Stopping JPEG capture and datachannel stream.")
            if self.is_jpeg_capturing and self.capture_module_instance:
                self.capture_module_instance.stop_capture()
                self.is_jpeg_capturing = False
                logger.info("JPEG capture stopped.")
            if self.jpeg_send_task:
                self.jpeg_send_task.cancel()
                self.jpeg_send_task = None
                logger.info("JPEG send task cancelled.")

        if self.data_channel:
            self.data_channel.emit('close')
            self.data_channel = None
            logger.info("data channel closed")
        if self.pipeline and self.render_mode == "gstreamer":
            logger.info("setting pipeline state to NULL")
            self.pipeline.set_state(Gst.State.NULL)
            self.pipeline = None
            logger.info("pipeline set to state NULL")
        if self.webrtcbin:
            self.webrtcbin.set_state(Gst.State.NULL)
            self.webrtcbin = None
            logger.info("webrtcbin set to state NULL")
        logger.info("pipeline stopped")


    class PlayoutDelayExtension(GstRtp.RTPHeaderExtension):
        def __init__(self):
            super().__init__()
            self.min_delay = 0
            self.max_delay = 0
            self.set_uri("http://www.webrtc.org/experiments/rtp-hdrext/playout-delay")

        def do_get_supported_flags(self):
            return GstRtp.RTPHeaderExtensionFlags.ONE_BYTE | GstRtp.RTPHeaderExtensionFlags.TWO_BYTE

        def do_get_max_size(self, input_meta):
            return 3

        def do_write(self, input_meta, write_flags, output, data, size):
            return 3

        def do_read(self, read_flags, data, size, buffer):
            self.min_delay = (data[0] << 4) | (data[1] >> 4)
            self.max_delay = ((data[1] & 0x0F) << 8) | data[2]
            return True
